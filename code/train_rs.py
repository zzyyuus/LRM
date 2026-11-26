"""Reproducible training/validation entry point. Test partition stays untouched."""
import argparse,json,random,time
from pathlib import Path
from collections import defaultdict
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
from PIL import Image
from our_models import resnet18_mamba_attention,resnet34_mamba_attention,resnet50_mamba_attention

class Records(Dataset):
    def __init__(self,records,transform):self.records,self.transform=records,transform
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        with Image.open(r['path']) as im:x=self.transform(im.convert('RGB'))
        return x,r['label']

def make_split(data,seed):
    rng=random.Random(seed); classes=sorted(data['classes']);split={k:[] for k in ('train','val','test')}
    for label,cls in enumerate(classes):
        groups=defaultdict(list)
        for r in data['records']:
            if r['class']==cls:groups[r['pixel_sha256']].append(dict(r,label=label))
        groups=list(groups.values());rng.shuffle(groups);groups.sort(key=len,reverse=True)
        n=data['classes'][cls];remaining={'train':int(.8*n),'val':int(.1*n),'test':n-int(.8*n)-int(.1*n)}
        for group in groups:
            choices=[k for k,v in remaining.items() if v>=len(group)]
            if not choices:raise ValueError('Cannot fit duplicate groups to exact split targets')
            k=rng.choices(choices,weights=[remaining[k] for k in choices])[0]
            split[k].extend(group);remaining[k]-=len(group)
        assert not any(remaining.values())
    hashes={k:{r['pixel_sha256'] for r in rows} for k,rows in split.items()}
    assert not(hashes['train']&hashes['val'] or hashes['train']&hashes['test'] or hashes['val']&hashes['test'])
    return dict(seed=seed,class_to_idx={c:i for i,c in enumerate(classes)},policy='new stratified 80/10/10; identical RGB pixel groups stay together; not recovered paper split',splits=split)

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--dataset',choices=['SIRI-WHU','UCMerced'],required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--backbone',type=int,choices=[18,34,50],default=18);p.add_argument('--context-dim',type=int,default=256);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--seed',type=int,default=42);p.add_argument('--split-seed',type=int,default=20260929);p.add_argument('--threads',type=int,default=4);p.add_argument('--device',default='cpu')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads)
    data=json.loads(Path(a.audit).read_text())[a.dataset]
    if data['bad']:raise ValueError('Unreadable images present')
    split=make_split(data,a.split_seed);(out/'split.json').write_text(json.dumps(split,indent=2))
    (out/'config.json').write_text(json.dumps(dict(vars(a),torch_version=torch.__version__,test_evaluated=False),indent=2))
    train_tf=T.Compose([T.RandomRotation(10),T.RandomHorizontalFlip(),T.Resize((240,240)),T.RandomCrop((224,224)),T.ToTensor(),T.Normalize([.5]*3,[.5]*3)])
    val_tf=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([.5]*3,[.5]*3)])
    train=DataLoader(Records(split['splits']['train'],train_tf),batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),num_workers=0)
    val=DataLoader(Records(split['splits']['val'],val_tf),batch_size=a.batch_size,num_workers=0)
    model={18:resnet18_mamba_attention,34:resnet34_mamba_attention,50:resnet50_mamba_attention}[a.backbone](num_classes=len(split['class_to_idx']),context_dim=a.context_dim).to(a.device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4);criterion=nn.CrossEntropyLoss();history=[];best=-1.;start=time.time()
    print(json.dumps(dict(dataset=a.dataset,counts={k:len(v) for k,v in split['splits'].items()},params=sum(p.numel() for p in model.parameters()),device=a.device)),flush=True)
    for epoch in range(1,a.epochs+1):
        model.train();loss_sum=correct=total=0
        for step,(x,y) in enumerate(train,1):
            x,y=x.to(a.device),y.to(a.device);opt.zero_grad(set_to_none=True);pred=model(x);loss=criterion(pred,y)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward()
            finite_grads = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
            if not finite_grads.item():raise RuntimeError('Non-finite gradient')
            opt.step();total+=len(y);loss_sum+=loss.item()*len(y);correct+=(pred.argmax(1)==y).sum().item()
            if step==1 or step%20==0:print(f'epoch={epoch} step={step}/{len(train)} loss={loss_sum/total:.4f} acc={correct/total:.4f} elapsed={time.time()-start:.1f}s',flush=True)
        model.eval();vl=vc=vn=0
        with torch.no_grad():
            for x,y in val:
                x,y=x.to(a.device),y.to(a.device);pred=model(x);loss=criterion(pred,y)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite validation loss')
                vn+=len(y);vl+=loss.item()*len(y);vc+=(pred.argmax(1)==y).sum().item()
        row=dict(epoch=epoch,train_loss=loss_sum/total,train_accuracy=correct/total,val_loss=vl/vn,val_accuracy=vc/vn,elapsed_seconds=time.time()-start)
        history.append(row);(out/'metrics.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
        if vc/vn>best:
            best=vc/vn;torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,config=vars(a),class_to_idx=split['class_to_idx']),out/'best.pt')
    print('Completed; test partition not evaluated.',flush=True)
if __name__=='__main__':main()
