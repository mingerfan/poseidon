"""B-owned original ResNet20 reference; no model-specific DSL lowering.

Architecture follows Yerlan Idelbayev's BSD-licensed CIFAR ResNet implementation,
revision d5489e8995e81e91ce6b1d69dcc98ad579b0b153. The downloaded implementation
is retained as data, never imported. Its notice is retained in UPSTREAM_LICENSE.
"""
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MEAN=(0.485,0.456,0.406)
STD=(0.229,0.224,0.225)
EPS=1e-5

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels,eps=EPS)
        self.conv2=nn.Conv2d(out_channels,out_channels,3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels,eps=EPS)
        self.stride=stride
        self.padding=out_channels//4 if stride==2 else 0
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        shortcut=x if self.stride==1 else F.pad(x[:,:,::2,::2],(0,0,0,0,self.padding,self.padding))
        return F.relu(out+shortcut)

class ResNet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,16,3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16,eps=EPS)
        self.layer1=nn.Sequential(BasicBlock(16,16,1),BasicBlock(16,16,1),BasicBlock(16,16,1))
        self.layer2=nn.Sequential(BasicBlock(16,32,2),BasicBlock(32,32,1),BasicBlock(32,32,1))
        self.layer3=nn.Sequential(BasicBlock(32,64,2),BasicBlock(64,64,1),BasicBlock(64,64,1))
        self.linear=nn.Linear(64,10)
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.layer3(self.layer2(self.layer1(out)))
        return self.linear(out.mean(dim=(2,3)))
    def forward_stages(self,x):
        stages=OrderedDict()
        x=F.relu(self.bn1(self.conv1(x)));stages['stem']=x
        for stage,layer in enumerate((self.layer1,self.layer2,self.layer3),1):
            for block,operation in enumerate(layer):
                x=operation(x);stages['layer%d.%d'%(stage,block)]=x
        x=x.mean(dim=(2,3));stages['mean']=x
        stages['logits']=self.linear(x)
        return stages

def load_original_weights(path):
    """Use the original restricted loader; no unrestricted pickle fallback."""
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if type(checkpoint)is not dict or not {'state_dict','best_prec1'}<=set(checkpoint):
        raise ValueError('Expected upstream checkpoint dictionary')
    raw=checkpoint['state_dict']
    if not isinstance(raw,(dict,OrderedDict))or not raw:raise ValueError('Checkpoint state mapping')
    if not all(type(k)is str and k.startswith('module.')for k in raw):raise ValueError('Checkpoint prefix')
    state=OrderedDict((k[len('module.'):],value)for k,value in raw.items())
    model=ResNet20().eval();expected=model.state_dict()
    absent=set(expected)-set(state)
    # Old BatchNorm snapshots predate this training-only integer counter.
    if any(not k.endswith('.num_batches_tracked')for k in absent):raise ValueError('Missing learned state')
    if set(state)-set(expected):raise ValueError('Unexpected checkpoint state')
    source_records={}
    for name,value in state.items():
        if not isinstance(value,torch.Tensor)or value.device.type!='cpu':raise ValueError('Tensor state required')
        if value.shape!=expected[name].shape or value.dtype!=expected[name].dtype:raise ValueError('Tensor shape/dtype')
        if not torch.isfinite(value).all().item():raise ValueError('Nonfinite tensor state')
        if name.endswith('running_var')and not (value>0).all().item():raise ValueError('Invalid running variance')
        source_records[name]=dict(shape=list(value.shape),dtype=str(value.dtype),elements=value.numel())
    result=model.load_state_dict(state,strict=True)
    if result.missing_keys or result.unexpected_keys:raise ValueError('Strict state load')
    for name,value in state.items():
        if not torch.equal(model.state_dict()[name],value):raise ValueError('Checkpoint values changed')
    return model,dict(original_state=source_records,default_training_only_counters=sorted(absent),
        parameter_elements=sum(p.numel()for p in model.parameters()),
        original_state_elements=sum(v.numel()for v in state.values()),
        checkpoint_metadata_keys=sorted(set(checkpoint)-{'state_dict'}),
        learned_weights_changed=False,restricted_weights_only=True)

def independent_stages(pixel_values,state):
    """Float64 NumPy convolution/indexing reference; never calls Torch or model."""
    values={name:np.asarray(value,dtype=np.float64)for name,value in state.items()}
    def conv(x,name,stride=1):
        weight=values[name+'.weight']
        padded=np.pad(x,((0,0),(0,0),(1,1),(1,1)))
        windows=np.lib.stride_tricks.sliding_window_view(padded,(3,3),axis=(2,3))[:,:,::stride,::stride,:,:]
        return np.einsum('nchwkl,ockl->nohw',windows,weight,optimize=True)
    def bn(x,name):
        shape=(1,-1,1,1)
        factor=values[name+'.weight']/np.sqrt(values[name+'.running_var']+EPS)
        return (x-values[name+'.running_mean'].reshape(shape))*factor.reshape(shape)+values[name+'.bias'].reshape(shape)
    x=np.asarray(pixel_values,dtype=np.float64)
    stages=OrderedDict()
    x=np.maximum(bn(conv(x,'conv1'),'bn1'),0);stages['stem']=x
    for stage,channels in enumerate((16,32,64),1):
        for block in range(3):
            name='layer%d.%d'%(stage,block);stride=2 if stage>1 and block==0 else 1
            shortcut=x if stride==1 else np.pad(x[:,:,::2,::2],((0,0),(channels//4,channels//4),(0,0),(0,0)))
            first=np.maximum(bn(conv(x,name+'.conv1',stride),name+'.bn1'),0)
            x=np.maximum(bn(conv(first,name+'.conv2'),name+'.bn2')+shortcut,0)
            stages[name]=x
    x=np.mean(x,axis=(2,3));stages['mean']=x
    stages['logits']=x@values['linear.weight'].T+values['linear.bias']
    return stages

def valid_domain_probes():
    """Four deterministic valid RGB inputs, unrelated to CIFAR accuracy."""
    values=np.empty((4,3,32,32),dtype=np.uint8)
    values[0]=0;values[1]=255
    checker=((np.indices((32,32)).sum(axis=0)%2)*255).astype(np.uint8)
    values[2]=checker
    grid=np.arange(3*32*32,dtype=np.uint16).reshape(3,32,32)
    values[3]=((grid*37+13)%256).astype(np.uint8)
    scaled=values.astype(np.float64)/255.0
    normalized=(scaled-np.asarray(MEAN).reshape(1,3,1,1))/np.asarray(STD).reshape(1,3,1,1)
    return values,normalized
