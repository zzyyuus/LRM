"""Portable launcher for the archived experiment runner."""
import argparse,json,subprocess,sys,tempfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--data-root',required=True,help='Directory containing class subdirectories')
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int)
    p.add_argument('--device')
    a=p.parse_args()
    base=Path(__file__).resolve().parent
    cfg=json.loads(Path(a.config).read_text())
    dataset=cfg['dataset'];split=base/'splits'/f'{dataset}_fold1.json'
    records=json.loads(split.read_text())
    missing=[r['path'] for part in records['splits'].values() for r in part if not(Path(a.data_root)/r['path']).is_file()]
    if missing:raise SystemExit(f'{len(missing)} missing images; first: {missing[0]}')
    if a.seed is not None:cfg['seed']=a.seed
    if a.device:cfg['device']=a.device
    # Explicit split supplies records; runner only needs the audit failure flag.
    with tempfile.TemporaryDirectory() as temp:
        audit=Path(temp)/'audit.json';audit.write_text(json.dumps({dataset:{'bad':[]}}))
        cmd=[sys.executable,str(base/'code/train_experiment.py'),'--audit',str(audit),'--split-file',str(split),'--data-root',str(Path(a.data_root).resolve()),'--output',a.output]
        for key,value in cfg.items():
            if value is None:continue
            flag='--'+key.replace('_','-')
            if isinstance(value,bool):
                if value:cmd.append(flag)
            else:cmd.extend([flag,str(value)])
        subprocess.run(cmd,check=True)
if __name__=='__main__':main()
