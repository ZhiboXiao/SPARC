"""Run SPARC on either bundled full-record sample; derive sparse input internally."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from common import ROOT,Sample,config,load_model,waveform_metrics
from model import cascade_tensor_native150
from windows import WindowContract,build_window_plan,OverlapAccumulator

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['simulation','real'],default='simulation')
    p.add_argument('--sample',type=Path,help='Optional replacement in the bundled HDF5 format')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'weights/sparc.pt')
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu');p.add_argument('--column-batch',type=int,default=6)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--start',type=int,default=0)
    p.add_argument('--length',type=int,help='Optional short interval for a fast smoke run; default is full record')
    p.add_argument('--output',type=Path);a=p.parse_args();out=a.output or ROOT/'outputs'/a.case
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    torch.set_num_threads(a.threads);sample=Sample(a.sample or ROOT/config()['cases'][a.case]);device=torch.device(a.device)
    if a.length is None and a.start!=0:raise ValueError('--start requires --length')
    length=sample.nt if a.length is None else a.length
    if not 40<=length<=sample.nt:raise ValueError('Invalid inference interval length')
    target=sample.grid(a.start,length);idx=sample.indices
    sparse=np.zeros_like(target);sparse[np.ix_(idx,idx)]=target[np.ix_(idx,idx)]
    if a.length is None:plan=sample.plan()
    else:
        contract=WindowContract(length=min(2560,length),nominal_stride=min(1280,length))
        plan=build_window_plan(sparse[np.ix_(idx,idx)].reshape(-1,length)/sample.scale,contract)
    model=load_model(a.checkpoint,device).eval();acc=OverlapAccumulator(sparse.shape,plan,dtype=np.float32);tick=time.perf_counter()
    for k,start in enumerate(plan.starts):
        segment=torch.from_numpy(np.ascontiguousarray(sparse[...,start:start+plan.contract.length]/sample.scale)).to(device)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            pred=cascade_tensor_native150(model,segment,column_batch=a.column_batch,known_indices=idx)
        acc.add(k,pred.float().cpu().numpy()*sample.scale)
        print(f'{sample.case}: window {k+1}/{len(plan.starts)}',flush=True)
    completed=acc.finish();completed[np.ix_(idx,idx)]=target[np.ix_(idx,idx)]
    report={'case':sample.case,'shape_xyt':list(completed.shape),'fs_hz':sample.fs,'source_interval':[a.start,a.start+length],
            'parameters':sum(p.numel() for p in model.parameters()),'scale':sample.scale,'device':str(device),
            'seconds':time.perf_counter()-tick,'full_record':a.length is None,**waveform_metrics(completed,target,idx)}
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'completion.npz',grid=completed,known_indices=idx,fs_hz=sample.fs,source_start=a.start)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));print(out)

if __name__=='__main__':main()
