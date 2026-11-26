"""Resumable experiment runner with validation-only early stopping."""
import argparse,json,math,random,time,os,sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from train_rs import Records,make_split
from experiment_models import build_model
from targeted_mixup import targeted_mix, pair_mix, mixup_batch

def write_json(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));temp.replace(path)

def save(path,obj):
    temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)

def scheduled_lr(base_lr,epoch,total,scheduler):
    """按 epoch 号直接算出学习率（无状态），保证断点续训与连续训练一致。"""
    if scheduler=='cosine':
        return base_lr*0.5*(1.0+math.cos(math.pi*(epoch-1)/max(1,total)))
    return base_lr

def resolve_device(name):
    """把 --device 解析成实际可用的设备：支持 cuda / cuda:N / mps / cpu / auto。

    auto（默认）优先 CUDA，其次 MPS，最后 CPU；显式传参时行为与以前一致。
    """
    if name=='auto':
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    if name.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        return name
    if name=='mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; run outside sandbox')
        return name
    return 'cpu'

def rng_state(device):
    if device.startswith('cuda'):
        return torch.cuda.get_rng_state()
    if device=='mps':
        return torch.mps.get_rng_state()
    return None

def set_rng_state(device,state):
    if state is None:
        return
    if device.startswith('cuda'):
        torch.cuda.set_rng_state(state)
    elif device=='mps':
        torch.mps.set_rng_state(state)

def make_train_transform(profile):
    """训练增强档位。

    paper = 论文原协议（±10° 旋转 + 水平翻转 + Resize240/Crop224），逐位保持原行为；
    geo   = 在 paper 基础上加完整二面体变换（90° 旋转 + 垂直翻转）+ 温和尺度抖动
            + 轻度颜色抖动，针对 UCMerced 的"尺度/密度/朝向"类错误。
    """
    if profile=='paper':
        return T.Compose([T.RandomRotation(10),T.RandomHorizontalFlip(),T.Resize((240,240)),
                          T.RandomCrop(224),T.ToTensor(),T.Normalize([.5]*3,[.5]*3)])
    if profile=='geo':
        return T.Compose([T.Resize((256,256)),
                          T.RandomHorizontalFlip(),T.RandomVerticalFlip(),
                          T.RandomApply([T.RandomRotation((90,90))],p=0.5),   # 与两次翻转合成完整 D4
                          T.RandomResizedCrop(224,scale=(0.7,1.0),ratio=(0.9,1.1)),
                          T.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
                          T.ToTensor(),T.Normalize([.5]*3,[.5]*3)])
    raise SystemExit('unknown aug-profile: '+profile)

@torch.no_grad()
def evaluate(model,loader,device,classes):
    model.eval();cm=torch.zeros(classes,classes,dtype=torch.long);loss_sum=n=0
    for x,y in loader:
        x,y=x.to(device),y.to(device);logits=model(x);loss=nn.functional.cross_entropy(logits,y)
        if not torch.isfinite(loss).item():raise RuntimeError('Nonfinite evaluation loss')
        pred=logits.argmax(1);cm+=torch.bincount((y*classes+pred).cpu(),minlength=classes*classes).reshape(classes,classes)
        loss_sum+=loss.item()*len(y);n+=len(y)
    tp=cm.diag().double();true=cm.sum(1).double();pred=cm.sum(0).double();acc=tp.sum().item()/n
    f1=(2*tp/(true+pred).clamp_min(1)).mean().item();pe=(true*pred).sum().item()/n**2
    return dict(loss=loss_sum/n,accuracy=acc,macro_f1=f1,kappa=(acc-pe)/(1-pe) if pe<1 else 0,confusion_matrix=cm.tolist())

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',help='JSON training configuration; command-line options override it')
    for key in ('audit','dataset','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--method',default='ours');p.add_argument('--depth',type=int,default=18);p.add_argument('--width',type=int,default=256)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--split-seed',type=int,default=20260929);p.add_argument('--device',default='auto');p.add_argument('--epochs',type=int,default=300);p.add_argument('--min-epochs',type=int,default=50);p.add_argument('--patience',type=int,default=30);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--threads',type=int,default=4);p.add_argument('--workers',type=int,default=0);p.add_argument('--resume-from')
    p.add_argument('--split-file')
    p.add_argument('--augmentation',choices=['none','hw_mixup','label_smoothing','pair_mixup','mixup'],default='none')
    p.add_argument('--label-smoothing',type=float,default=0.05)
    # 混合增强参数：hw_mixup / pair_mixup 是定向配对，mixup 是全类标准 Mixup。
    # 默认值与原先写死的取法完全一致（0.25 / Beta(0.2,0.2)），保证旧作业口径不变。
    p.add_argument('--mixup-pair',default='harbor,water',help='pair_mixup 使用的两类，逗号分隔')
    p.add_argument('--mixup-probability',type=float,default=0.25)
    p.add_argument('--mixup-alpha',type=float,default=0.2)
    p.add_argument('--aug-profile',choices=['paper','geo'],default='paper',
                   help='paper=论文原增强；geo=加二面体变换/尺度抖动/颜色抖动')
    p.add_argument('--data-root',default='',
                   help='数据集根目录；划分文件里的相对路径会拼到它下面（便于迁移到服务器）')
    p.add_argument('--ema-decay',type=float,default=0.0)   # >0 时启用权重指数滑动平均
    # 训练稳定性相关：当前实验默认 (lr=1e-4, weight_decay=0, scheduler=none, min_delta=0)
    # 与原协议完全一致；显式传参才启用正则化/学习率衰减，避免旧结果口径被悄悄改变。
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=0.0)
    p.add_argument('--scheduler',choices=['none','cosine'],default='none')
    p.add_argument('--min-delta',type=float,default=0.0,
                   help='早停判据至少需要提升这么多才重置 patience，用于过滤小验证集上的抖动。')
    p.add_argument('--select',default='val_best',choices=['val_best','ema_best','ema_best_loss','last'])
    # 早停判据：默认与选权重共用同一指标（有 EMA 时用 EMA 验证），避免
    # 「按普通验证早停、却按 EMA 验证选权重」两个判据不一致。
    p.add_argument('--early-stop-metric',default=None,choices=['val','ema_val','ema_val_loss'])
    p.add_argument('--validation-only',action='store_true',
                   help='Finish after validation and save no test metrics; use for model development.')
    p.add_argument('--test-selected-only',action='store_true',
                   help='Evaluate only the predeclared selected checkpoint on test.')
    probe=argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config')
    config_path=probe.parse_known_args()[0].config
    config_args=[]
    if config_path:
        try:
            settings=json.loads(Path(config_path).read_text(encoding='utf-8'))
        except (OSError,ValueError) as exc:
            p.error('Cannot read config: '+str(exc))
        if not isinstance(settings,dict):p.error('Config must be a JSON object')
        actions={action.dest:action for action in p._actions}
        for key,value in settings.items():
            if key not in actions or key in ('config','help'):p.error('Unknown config key: '+key)
            action=actions[key]
            if value is None:continue
            if isinstance(action,argparse._StoreTrueAction):
                if not isinstance(value,bool):p.error(key+' must be boolean')
                if value:config_args.append(action.option_strings[0])
            else:
                if isinstance(value,(dict,list,bool)):p.error('Invalid config value: '+key)
                config_args.extend([action.option_strings[0],str(value)])
    a=p.parse_args(config_args+sys.argv[1:]);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    # 早停判据默认跟随「选权重规则」，保证两者永远是同一个量：
    #   select=val_best -> 普通验证准确率；select=ema_best -> EMA 验证准确率；
    #   select=ema_best_loss -> EMA 验证损失（对过拟合更敏感）；select=last -> EMA 验证准确率。
    if a.early_stop_metric is None:
        a.early_stop_metric={'val_best':'val','ema_best':'ema_val','ema_best_loss':'ema_val_loss'}.get(a.select,'ema_val')
    if a.early_stop_metric.startswith('ema_') and a.ema_decay<=0:
        raise SystemExit('早停判据为 {}（select={}）需要 --ema-decay > 0'.format(a.early_stop_metric,a.select))
    if (out/'result.json').exists():print('Already completed',out,flush=True);return
    a.device=resolve_device(a.device)
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);random.seed(a.seed)
    data=json.loads(Path(a.audit).read_text())[a.dataset]
    if data['bad']:raise RuntimeError('Dataset decode failures')
    if a.split_file:
        # 外部划分（例如防泄漏的 v2 划分）：直接采用，并记录来源以便追溯。
        split_source=str(Path(a.split_file).resolve())
        split=json.loads(Path(split_source).read_text())
        for part in ('train','val','test'):
            for record in split['splits'][part]:
                if not isinstance(record['label'],int):
                    raise RuntimeError('外部划分的 label 必须是整数索引：'+str(record['label'])[:40])
                if a.data_root and not Path(record['path']).is_absolute():
                    record['path']=str(Path(a.data_root)/record['path'])
    else:
        split_source='make_split(split_seed=%d)'%a.split_seed
        split=make_split(data,a.split_seed)
    if (out/'split.json').exists() and json.loads((out/'split.json').read_text())!=split:raise RuntimeError('Split changed')
    write_json(out/'split.json',split)
    cfg=vars(a)|{'evaluation_role':'development_exposed_test','torch_version':torch.__version__,'early_stop_monitor':a.early_stop_metric,'min_delta':a.min_delta,'pretrained':False,'lr':a.lr,'weight_decay':a.weight_decay,'scheduler':None if a.scheduler=='none' else a.scheduler,'split_source':split_source}
    if (out/'config.json').exists():
        old=json.loads((out/'config.json').read_text())
        for key in ('dataset','method','depth','width','seed','split_seed','batch_size','patience','min_epochs','augmentation','label_smoothing'):
            if old[key]!=cfg[key]:raise RuntimeError('Config changed: '+key)
    write_json(out/'config.json',cfg)
    tf=make_train_transform(a.aug_profile)
    vf=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([.5]*3,[.5]*3)])
    generator=torch.Generator().manual_seed(a.seed)
    if a.workers<0:raise RuntimeError('workers must be >= 0')
    persistent=a.workers>0
    train=DataLoader(Records(split['splits']['train'],tf),batch_size=a.batch_size,shuffle=True,generator=generator,num_workers=a.workers,persistent_workers=persistent)
    val=DataLoader(Records(split['splits']['val'],vf),batch_size=a.batch_size,num_workers=a.workers,persistent_workers=persistent)
    classes=len(split['class_to_idx']);model=build_model(a.method,a.depth,classes,a.width).to(a.device);opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    history=[];best=-1.;bad_epochs=0;first=1;elapsed_base=0;resume_note=None;best_monitored=-1.
    resumed_ema_state=None;resumed_best_ema=None;resumed_best_ema_loss=None
    resume=out/'last.pt' if (out/'last.pt').exists() else Path(a.resume_from) if a.resume_from else None
    if resume:
        cp=torch.load(resume,map_location='cpu',weights_only=False)
        if cp['class_to_idx']!=split['class_to_idx']:raise RuntimeError('Class mapping mismatch')
        model.load_state_dict(cp['model'],strict=True);opt.load_state_dict(cp['optimizer']);first=cp['epoch']+1
        if 'history' in cp:
            resume_note=cp.get('resume_note');history=cp['history'];best=cp['best'];bad_epochs=cp['bad_epochs'];elapsed_base=cp['elapsed']
            best_monitored=cp.get('best_monitored',-1.)
            if a.ema_decay>0:
                if 'ema_state' not in cp or 'best_ema' not in cp:
                    raise RuntimeError('Cannot faithfully resume EMA: checkpoint lacks ema_state/best_ema. '
                                       'The previous best_ema.pt must not be overwritten.')
                resumed_ema_state=cp['ema_state'];resumed_best_ema=cp['best_ema'];resumed_best_ema_loss=cp.get('best_ema_loss')
            torch.set_rng_state(cp['rng_torch']);random.setstate(cp['rng_python']);generator.set_state(cp['rng_loader'])
            set_rng_state(a.device,cp.get('rng_mps') or cp.get('rng_device'))
        else:
            prior_split=resume.parent/'split.json'
            if json.loads(prior_split.read_text())!=split:raise RuntimeError('Historical checkpoint split mismatch')
            prior=json.loads((resume.parent/'metrics.json').read_text())
            best=max(r['val_accuracy'] for r in prior);resume_note='Continued one-epoch checkpoint; historical RNG states unavailable, reseeded continuation (not bitwise uninterrupted).'
            save(out/'best.pt',dict(model=cp['model'],epoch=cp['epoch'],class_to_idx=split['class_to_idx']))
        del cp
    if best_monitored < 0 and history and a.early_stop_metric=='ema_val':
        # 兼容旧检查点：从历史里恢复该判据的历史最优值，避免误判为「刚开始」。
        values=[r['ema_val_accuracy'] for r in history if r.get('ema_val_accuracy') is not None]
        if values:best_monitored=max(values)
    if best_monitored < 0 and history and a.early_stop_metric=='ema_val_loss':
        # 同上，但判据变成 EMA 验证损失：内部统一为「越大越好」，所以取负号。
        values=[r['ema_val_loss'] for r in history if r.get('ema_val_loss') is not None]
        if values:best_monitored=-min(values)
    params=sum(p.numel() for p in model.parameters());start=time.time()
    # 权重指数滑动平均（EMA）：小验证集上「按最佳验证值挑权重」噪声很大，
    # EMA 权重通常更稳。ema_state 只在 ema_decay>0 时维护。
    ema_state=None;best_ema=-1.0;best_ema_loss=float('inf')
    if a.ema_decay>0:
        ema_state=({k:v.to(a.device) for k,v in resumed_ema_state.items()} if resumed_ema_state is not None else
                   {k:(v.detach().clone().float() if v.is_floating_point() else v.detach().clone())
                    for k,v in model.state_dict().items()})
        best_ema=resumed_best_ema if resumed_best_ema is not None else -1.0
        if resumed_best_ema_loss is not None:best_ema_loss=resumed_best_ema_loss
    print(json.dumps(dict(event='start',method=a.method,dataset=a.dataset,depth=a.depth,width=a.width,seed=a.seed,first_epoch=first,params=params,resume_note=resume_note)),flush=True)
    reason='max_epochs'
    for epoch in range(first,a.epochs+1):
        for group in opt.param_groups:group['lr']=scheduled_lr(a.lr,epoch,a.epochs,a.scheduler)
        model.train();loss_sum=correct=total=0
        mix_rng=random.Random(a.seed+1000003*epoch);mixed=eligible=0
        for x,y in train:
            x,y=x.to(a.device),y.to(a.device);opt.zero_grad(set_to_none=True)
            yb=y;lam=torch.ones(len(y),device=a.device)
            if a.augmentation in ('hw_mixup','pair_mixup','mixup'):
                if a.augmentation=='mixup':
                    x,yb,lam,e,n=mixup_batch(x,y,mix_rng,a.mixup_probability,a.mixup_alpha)
                else:
                    names=[v.strip() for v in a.mixup_pair.split(',')]
                    if len(names)!=2:raise SystemExit('--mixup-pair 需要恰好两个类别名')
                    missing=[v for v in names if v not in split['class_to_idx']]
                    if missing:raise SystemExit(f'{a.dataset} 里没有类别 {missing}，无法做定向混合')
                    x,yb,lam,e,n=pair_mix(x,y,split['class_to_idx'][names[0]],
                                          split['class_to_idx'][names[1]],mix_rng,
                                          a.mixup_probability,a.mixup_alpha)
                eligible+=e;mixed+=n
            pred=model(x)
            if a.augmentation in ('hw_mixup','pair_mixup','mixup'):
                loss=(lam*nn.functional.cross_entropy(pred,y,reduction='none')+(1-lam)*nn.functional.cross_entropy(pred,yb,reduction='none')).mean()
            else:
                loss=nn.functional.cross_entropy(pred,y,label_smoothing=a.label_smoothing if a.augmentation=='label_smoothing' else 0.)
            if not torch.isfinite(loss).item():raise RuntimeError('Nonfinite loss')
            loss.backward()
            if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all().item():raise RuntimeError('Nonfinite gradients')
            opt.step();total+=len(y);correct+=(lam*(pred.argmax(1)==y)+(1-lam)*(pred.argmax(1)==yb)).sum().item();loss_sum+=loss.item()*len(y)
            if ema_state is not None:
                with torch.no_grad():
                    for key,value in model.state_dict().items():
                        if value.is_floating_point():
                            ema_state[key].mul_(a.ema_decay).add_(value.detach().float(),alpha=1-a.ema_decay)
                        else:
                            ema_state[key].copy_(value)
        metrics=evaluate(model,val,a.device,classes);score=metrics['accuracy']
        if score>best:
            best=score;save(out/'best.pt',dict(model=model.state_dict(),epoch=epoch,class_to_idx=split['class_to_idx']))
        ema_score=None
        if ema_state is not None:
            backup={k:v.detach().clone() for k,v in model.state_dict().items()}
            model.load_state_dict({k:v.to(backup[k].dtype) for k,v in ema_state.items()},strict=True)
            ema_metrics=evaluate(model,val,a.device,classes);ema_score=ema_metrics['accuracy']
            if ema_score>best_ema:
                best_ema=ema_score;save(out/'best_ema.pt',dict(model=model.state_dict(),epoch=epoch,class_to_idx=split['class_to_idx']))
            if ema_metrics['loss']<best_ema_loss:
                best_ema_loss=ema_metrics['loss'];save(out/'best_ema_loss.pt',dict(model=model.state_dict(),epoch=epoch,class_to_idx=split['class_to_idx']))
            model.load_state_dict(backup,strict=True)
        # 统一判据：早停计数跟随 a.early_stop_metric，与选权重使用同一个量；
        # 另一个指标仍逐轮记录，但只作诊断，不参与任何决策。
        if a.early_stop_metric=='ema_val_loss':
            monitored=-ema_metrics['loss'];monitored_value=ema_metrics['loss']
        elif a.early_stop_metric=='ema_val' and ema_score is not None:
            monitored=ema_score;monitored_value=ema_score
        else:
            monitored=score;monitored_value=score
        if monitored>best_monitored+a.min_delta:
            best_monitored=monitored;bad_epochs=0
        else:
            bad_epochs+=1
        row=dict(epoch=epoch,train_loss=loss_sum/total,train_accuracy=correct/total,val_loss=metrics['loss'],val_accuracy=score,val_macro_f1=metrics['macro_f1'],best_val_accuracy=best,bad_epochs=bad_epochs,early_stop_metric=a.early_stop_metric,monitored_metric=a.early_stop_metric,monitored_value=monitored_value,best_monitored=best_monitored,best_monitored_accuracy=best_monitored,learning_rate=opt.param_groups[0]['lr'],elapsed_seconds=elapsed_base+time.time()-start)
        if ema_score is not None:row['ema_val_accuracy']=ema_score;row['best_ema_val_accuracy']=best_ema
        row.update(mixed=mixed,mix_eligible=eligible,train_accuracy_kind='soft_target_agreement' if a.augmentation in ('hw_mixup','pair_mixup','mixup') else 'hard_target_accuracy',val_confusion_matrix=metrics['confusion_matrix'])
        if ema_score is not None:row.update(ema_val_loss=ema_metrics['loss'],ema_val_macro_f1=ema_metrics['macro_f1'],ema_val_confusion_matrix=ema_metrics['confusion_matrix'])
        history.append(row);write_json(out/'metrics.json',history);print(json.dumps(row),flush=True)
        cp=dict(resume_note=resume_note,model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,class_to_idx=split['class_to_idx'],history=history,best=best,bad_epochs=bad_epochs,best_monitored=best_monitored,elapsed=row['elapsed_seconds'],rng_torch=torch.get_rng_state(),rng_python=random.getstate(),rng_loader=generator.get_state(),rng_mps=rng_state(a.device),ema_state=ema_state,best_ema=best_ema,best_ema_loss=best_ema_loss)
        save(out/'last.pt',cp);del cp
        if epoch>=a.min_epochs and bad_epochs>=a.patience:reason='early_stopping';break
    if a.validation_only:
        result=dict(config=cfg,params=params,best_val_accuracy=best,
                    best_ema_val_accuracy=(best_ema if ema_state is not None else None),
                    best_ema_val_loss=(best_ema_loss if ema_state is not None else None),
                    stop_reason=reason,last_epoch=epoch,select=a.select,
                    validation_only=True,ema_resume_safe=True,resume_note=resume_note)
        write_json(out/'result.json',result)
        print(json.dumps(dict(event='validation_completed',output=str(out),
                              best_val_accuracy=best,best_ema_val_accuracy=result['best_ema_val_accuracy'])),flush=True)
        return
    test=DataLoader(Records(split['splits']['test'],vf),batch_size=a.batch_size,num_workers=0)
    if a.test_selected_only:
        checkpoint={'val_best':'best.pt','ema_best':'best_ema.pt','ema_best_loss':'best_ema_loss.pt','last':'last.pt'}[a.select]
        model.load_state_dict(torch.load(out/checkpoint,map_location='cpu',weights_only=True)['model'])
        primary=evaluate(model,test,a.device,classes)
        result=dict(config=cfg,params=params,best_val_accuracy=best,
                    best_ema_val_accuracy=best_ema,best_ema_val_loss=best_ema_loss,stop_reason=reason,last_epoch=epoch,
                    select=a.select,test=primary,test_selection={a.select:primary},
                    ema_resume_safe=True,resume_note=resume_note)
        write_json(out/'result.json',result)
        return
    # 同一批训练权重，用三种选权重方式各评一次测试集：只有 select 指定的那个作为主指标，
    # 其余仅作为「选权重方式影响多大」的诊断记录，不参与任何选择。
    variants={}
    variants['last']=evaluate(model,test,a.device,classes)
    if (out/'best.pt').exists():
        model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=True)['model'])
        variants['val_best']=evaluate(model,test,a.device,classes)
    if (out/'best_ema.pt').exists():
        model.load_state_dict(torch.load(out/'best_ema.pt',map_location='cpu',weights_only=True)['model'])
        variants['ema_best']=evaluate(model,test,a.device,classes)
    primary=variants.get(a.select,variants['last'])
    result=dict(config=cfg,params=params,best_val_accuracy=best,best_ema_val_accuracy=(best_ema if ema_state is not None else None),stop_reason=reason,last_epoch=epoch,select=a.select,test=primary,test_selection=variants,ema_resume_safe=True,resume_note=resume_note)
    write_json(out/'result.json',result);print(json.dumps(dict(event='completed',output=str(out),test=result['test']['accuracy'])),flush=True)
if __name__=='__main__':main()
