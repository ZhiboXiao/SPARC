"""Single-simulation clean training demo; not a full dataset performance study."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import ROOT,config,Sample,load_model,clean_batch,loss

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample',type=Path,default=ROOT/'data/simulation.h5')
    p.add_argument('--initialize',type=Path,help='Optional continuation; omitted means random initialization')
    p.add_argument('--steps',type=int,default=10);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--time-size',type=int,default=2560);p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--output',type=Path,default=ROOT/'outputs/train_demo.pt');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sample=Sample(a.sample);cfg=config()
    if sample.case!='simulation':raise ValueError('The real sample is for inference only')
    if a.steps<1 or not 40<=a.time_size<=2560:raise ValueError('Use positive steps and time-size in [40,2560]')
    torch.set_num_threads(a.threads);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);device=torch.device(a.device)
    model=load_model(a.initialize,device);model.train()
    opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':cfg['backbone_lr']},
            {'params':model.temporal_refiner.parameters(),'lr':cfg['refiner_lr']}],weight_decay=cfg['weight_decay'])
    assert all(p.requires_grad for p in model.parameters())
    plan=sample.plan();history=[]
    for step in range(1,a.steps+1):
        start,_=plan.sample_start(rng)
        if a.time_size<2560:start+=int(rng.integers(0,2560-a.time_size+1))
        x,y=clean_batch(sample,start,a.time_size,a.batch_size,rng,device);opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            value=loss(model(x),y,cfg['cone_weight'])
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite loss')
        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        record={'step':step,'loss':float(value.detach()),'start':start};history.append(record);print(json.dumps(record),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'method':'SPARC','step':a.steps,
                'role':'one-sample training demonstration, not reference weights','initialization':'random' if a.initialize is None else 'checkpoint'},a.output)
    a.output.with_suffix('.json').write_text(json.dumps({'history':history,'batch_size':a.batch_size,'time_size':a.time_size,'all_parameters_trainable':True},indent=2)+'\n')
    print(a.output)

if __name__=='__main__':main()
