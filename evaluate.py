"""Evaluate a checkpoint produced by this repository."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'code'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root',required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from train_rs import Records
    from train_experiment import evaluate,resolve_device
    from experiment_models import build_model
    cfg=json.loads(Path(a.config).read_text());device=resolve_device(a.device)
    split=json.loads((Path(__file__).resolve().parent/'splits'/f"{cfg['dataset']}_fold1.json").read_text())
    for r in split['splits']['test']:r['path']=str(Path(a.data_root)/r['path'])
    model=build_model(cfg['method'],cfg['depth'],len(split['class_to_idx']),cfg['width']).to(device)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state['model'])
    tf=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([.5]*3,[.5]*3)])
    loader=DataLoader(Records(split['splits']['test'],tf),batch_size=cfg['batch_size'],shuffle=False)
    print(json.dumps(evaluate(model,loader,device,len(split['class_to_idx'])),indent=2))
if __name__=='__main__':main()
