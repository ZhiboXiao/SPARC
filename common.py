"""Shared two-sample I/O, clean training batches and waveform objectives."""
from pathlib import Path
import json
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from model import Native150PolyphaseUNet
from windows import WindowPlan,WindowContract,build_window_plan

ROOT=Path(__file__).resolve().parent

def config():
    return json.loads((ROOT/'config.json').read_text())

class Sample:
    def __init__(self,path):
        self.path=Path(path)
        with h5py.File(self.path,'r') as f:
            self.size=int(f.attrs['spatial_size']);self.nt=f['signal'].shape[1]
            self.fs=float(f.attrs['fs_hz']);self.scale=float(f.attrs['scale'])
            self.case=str(f.attrs['case']);self.bounds=json.loads(f.attrs['active_bounds_json'])
            self.plan_dict=json.loads(f.attrs['window_plan_json'])
        if self.size!=102 or self.fs!=150e6 or not self.scale>0:
            raise ValueError('This demo uses the registered 102-grid / 150-MHz anatomy model')
        self.indices=np.arange(0,self.size,4)
    def rows(self,indices,start,length):
        if start<0 or start+length>self.nt:raise ValueError('Temporal interval exceeds the sample')
        with h5py.File(self.path,'r') as f:return np.asarray(f['signal'][indices,start:start+length],dtype='f4')
    def grid(self,start=0,length=None):
        length=self.nt-start if length is None else length
        return self.rows(np.arange(self.size**2),start,length).reshape(self.size,self.size,length).transpose(1,0,2)
    def plan(self):
        d=dict(self.plan_dict);d.pop('stops',None);d['contract']=WindowContract(**d['contract'])
        for key in ['starts','information_scores','normalized_information','fusion_priorities','sampling_probabilities','normalized_rms','median_signal_snr','occupancies']:
            d[key]=np.asarray(d[key],dtype='i8' if key=='starts' else 'f8')
        d['channel_noise_rms']=np.ones(len(self.indices)**2)
        return WindowPlan(**d)

def load_model(checkpoint,device):
    model=Native150PolyphaseUNet().to(device)
    if checkpoint is not None:
        cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(cp['model'],strict=True)
    return model

def clean_batch(sample,start,length,batch,rng,device):
    """Five neighbouring XT or YT sections; mask offset and spacing are sampled."""
    choices=[];lines=[]
    for _ in range(batch):
        axis=int(rng.integers(0,2));spacing=int(rng.choice((1,4)))
        offset=int(rng.integers(0,4));center=int(rng.integers(2*spacing,sample.size-2*spacing))
        neighbors=center+np.arange(-2,3)*spacing
        line=[np.arange(sample.size)+v*sample.size if axis==0 else np.arange(sample.size)*sample.size+v for v in neighbors]
        choices.append((offset,line));lines.extend(line)
    ids=np.unique(np.concatenate(lines));raw=sample.rows(ids,start,length)/sample.scale
    x=np.zeros((batch,10,sample.size,length),dtype='f4');y=np.empty((batch,1,sample.size,length),dtype='f4')
    for b,(offset,line) in enumerate(choices):
        mask=np.zeros((sample.size,length),dtype='f4');mask[offset::4]=1
        for k,l in enumerate(line):
            values=raw[np.searchsorted(ids,l)];x[b,2*k]=values*mask;x[b,2*k+1]=mask
            if k==2:y[b,0]=values
    x=torch.from_numpy(x).to(device);y=torch.from_numpy(y).to(device)
    pad=(-sample.size)%16
    return F.pad(x,(0,0,0,pad),mode='reflect'),F.pad(y,(0,0,0,pad),mode='reflect')

def loss(pred,target,cone_weight=.1):
    signal=F.l1_loss(pred,target)
    spatial=F.l1_loss(pred[:,:,1:]-pred[:,:,:-1],target[:,:,1:]-target[:,:,:-1])
    temporal=(150/29.8)*F.l1_loss(pred[...,1:]-pred[...,:-1],target[...,1:]-target[...,:-1])
    p=pred[:,0,:102].float();k=torch.fft.fftfreq(102,d=.5,device=p.device)
    f=torch.fft.fftfreq(p.shape[-1],d=1/150e6,device=p.device)
    outside=(k.abs()[:,None]>f.abs()[None,:]/1490e3).float()
    power=torch.fft.fft2(p,dim=(1,2)).abs().square();cone=(power*outside).sum()/(power.sum()+1e-9)
    return signal+.5*(spatial+temporal)+cone_weight*cone

def waveform_metrics(pred,target,indices):
    """Pooled unknown-position NRMSE and Pearson r, streamed across channels."""
    mask=np.ones(target.shape[:2],bool);mask[np.ix_(indices,indices)]=False;coords=np.argwhere(mask)
    count=sx=sy=sxx=syy=sxy=error=0.
    for j in range(0,len(coords),64):
        q=coords[j:j+64];x=pred[q[:,0],q[:,1]].astype('f8');y=target[q[:,0],q[:,1]].astype('f8')
        count+=x.size;sx+=x.sum();sy+=y.sum();sxx+=(x*x).sum();syy+=(y*y).sum();sxy+=(x*y).sum();error+=((x-y)**2).sum()
    vx=max(sxx-sx*sx/count,1e-30);vy=max(syy-sy*sy/count,1e-30)
    return {'unknown_nrmse':float(np.sqrt(error/vy)),'unknown_pearson_r':float((sxy-sx*sy/count)/np.sqrt(vx*vy)),
            'measured_max_abs_error':float(np.max(np.abs(pred[np.ix_(indices,indices)]-target[np.ix_(indices,indices)])))}
