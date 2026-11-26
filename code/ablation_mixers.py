import math
import torch
from torch import nn
from our_models import RMSNorm

def count(m):return sum(p.numel() for p in m.parameters())

class AttentionMixerMatched(nn.Module):
    def __init__(self,d,target):
        super().__init__();self.norm=RMSNorm(d);self.attn=nn.MultiheadAttention(d,4,batch_first=True)
        h=max(1,round((target-count(self)-d)/(2*d+1)))
        self.ffn=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Linear(h,d))
    def forward(self,x):
        z=self.norm(x);y=self.attn(z,z,z,need_weights=False)[0];return y+self.ffn(y)

class MLPMixerMatched(nn.Module):
    def __init__(self,d,target,tokens=49):
        super().__init__();self.norm=RMSNorm(d);self.token=nn.Sequential(nn.Linear(tokens,256),nn.GELU(),nn.Linear(256,tokens))
        h=max(1,round((target-count(self)-d)/(2*d+1)))
        self.channel=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Linear(h,d))
    def forward(self,x):
        z=self.norm(x);y=self.token(z.transpose(1,2)).transpose(1,2);return y+self.channel(y)

class LargeKernelMatched(nn.Module):
    def __init__(self,d,target):
        super().__init__();self.norm=RMSNorm(d);self.kernel=nn.Conv2d(d,d,7,padding=3,groups=d,bias=False)
        h=max(1,round((target-count(self)-d)/(2*d+1)))
        self.channel=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Linear(h,d))
    def forward(self,x):
        z=self.norm(x);b,n,d=z.shape;s=math.isqrt(n);assert s*s==n
        y=self.kernel(z.transpose(1,2).reshape(b,d,s,s)).flatten(2).transpose(1,2);return y+self.channel(y)

class SharedBidirectional(nn.Module):
    def __init__(self,block):super().__init__();self.block=block
    def forward(self,x):return .5*(self.block(x)+self.block(x.flip(1)).flip(1))

def replace_mixer(model,variant):
    att=model.mamba_attention;old=att.mamba_block;target=count(old);d=old.inp_proj.in_features
    if variant=='self_attention':att.mamba_block=AttentionMixerMatched(d,target)
    elif variant=='mlp':att.mamba_block=MLPMixerMatched(d,target)
    elif variant=='large_kernel':att.mamba_block=LargeKernelMatched(d,target)
    elif variant=='nonselective_ssm':old.S6.to_nonselective()
    elif variant=='bidirectional':att.mamba_block=SharedBidirectional(old)
    else:raise ValueError(variant)
    return model
