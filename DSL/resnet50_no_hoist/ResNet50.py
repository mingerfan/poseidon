#!/usr/bin/env python3
"""Model-level ResNet50; preserve the validated native maxpool polynomial."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import imagenet_layout as layout

IMAGE_SIZE = 224
IMAGE_VALUES = 3 * IMAGE_SIZE * IMAGE_SIZE

def load_image(values, image_id=0):
    need = (image_id + 1) * IMAGE_VALUES
    with values.open() as stream:
        data = np.fromiter(
            (float(token) for line in stream for token in line.split()),
            dtype=np.float64,
            count=need,
        )
    if data.size != need or not np.isfinite(data).all():
        raise ValueError("missing ImageNet image values")
    return torch.from_numpy(data[-IMAGE_VALUES:].reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE))

def load_label(labels, image_id=0):
    with labels.open() as stream:
        values = [int(token) for line in stream for token in line.split()]
    return values[image_id]

def component(tree_rows, coefficients, degree, baby_length, divisor, scalar):
    rows = [[int(v) for v in row.split()] for row in tree_rows]
    depth = len(rows) - 1
    nodes = [-1] + [v for row in rows for v in row]
    size = 1 << (depth + 1)
    nodes += [-1] * (size - len(nodes))
    degrees, starts = [-1] * size, [-1] * size
    degrees[1] = degree
    for level in range(1, depth + 1):
        for node in range(1 << level, 1 << (level + 1)):
            degrees[node] = (
                nodes[node // 2] - 1
                if node % 2 == 0
                else degrees[node // 2] - nodes[node // 2]
            )
    cursor = 1
    for node in range(1, size):
        if nodes[node] == 0:
            starts[node] = cursor
            cursor += degrees[node] + 1
    if cursor - 1 != len(coefficients):
        raise ValueError("d13 coefficient count does not match polynomial tree")
    coeff = [scalar(float(c) / divisor) for c in coefficients]
    two, minus_one = scalar(2.0), scalar(-1.0)

    def evaluate(x):
        giant = {1: x}

        def power(n):
            if n not in giant:
                half = power(n // 2)
                giant[n] = two * half * half + minus_one
            return giant[n]

        baby = [x]
        for level in range(1, baby_length.bit_length() - 1):
            products = [two * value * power(1 << level) for value in baby]
            baby += [a - b for a, b in zip(products, reversed(baby))]

        def subtree(node):
            if nodes[node] == 0:
                start = starts[node]
                terms = [
                    coeff[start + 2 * k] * baby[k]
                    for k in range((degrees[node] + 1) // 2)
                ]
                result = terms[0]
                for term in terms[1:]:
                    result = result + term
                return result
            return subtree(2 * node + 1) * power(nodes[node]) + subtree(2 * node)

        return subtree(1)

    return evaluate

def make_relu(coefficients, scalar):
    from poly import Poly

    values = np.loadtxt(coefficients, dtype=np.float64).reshape(-1)
    if values.size != 60 or not np.isfinite(values).all():
        raise ValueError("expected all 60 finite original d13 coefficients")
    p1 = component(Poly.treeStr1, values[:16], 15, 4, 2.0, scalar)
    p2 = component(Poly.treeStr2, values[16:32], 15, 4, 1.7, scalar)
    p3 = component(Poly.treeStr3, values[32:], 27, 8, 2.0, scalar)
    half = scalar(0.5)
    return lambda x: x * (p3(p2(p1(x))) + half)

SLOTS = 32768
BOUNDARY = 120.0
INPUT_PACKS = 8


class Bottleneck(nn.Module):
    def __init__(self,channels,planes,stride):
        super().__init__()
        self.conv1 = nn.Conv2d(channels,planes,1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,3,stride,1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,planes*4,1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.downsample = None
        if stride != 1 or channels != planes*4:
            self.downsample = nn.Sequential(nn.Conv2d(channels,planes*4,1,stride,bias=False),
                                            nn.BatchNorm2d(planes*4))


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,7,2,3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        channels = 64
        for stage,(planes,count) in enumerate(zip((64,128,256,512),(3,4,6,3)),1):
            blocks = []
            for block in range(count):
                blocks.append(Bottleneck(channels,planes,2 if stage>1 and block==0 else 1))
                channels = planes*4
            setattr(self,f'layer{stage}',nn.Sequential(*blocks))
        self.fc = nn.Linear(2048,1000)


def load_model(weights):
    model = ResNet50().double().eval()
    files = {p.name for p in weights.glob('*.txt')}
    if len(files) != 267:
        raise ValueError(f'expected 267 weights, found {len(files)}')
    state,matched = model.state_dict(),set()
    for name,tensor in state.items():
        filename = name.replace('.','_')+'.txt'
        if filename not in files:
            if name.endswith('num_batches_tracked'):
                continue
            raise ValueError(f'missing weight {filename}')
        values = np.asarray(np.loadtxt(weights/filename,dtype=np.float64))
        if values.size != tensor.numel() or not np.isfinite(values).all():
            raise ValueError(f'invalid weight {filename}')
        state[name] = torch.from_numpy(values.copy()).reshape(tensor.shape).to(tensor.dtype)
        matched.add(filename)
    if files != matched:
        raise ValueError(f'unmapped weights {files-matched}')
    model.load_state_dict(state,strict=True)
    return model


def make_pool_relu(coefficients,scalar):
    from poly import Poly
    values = np.loadtxt(coefficients,dtype=np.float64).reshape(-1)
    if values.size != 60 or not np.isfinite(values).all():
        raise ValueError('expected original 60 d13 coefficients')
    p1 = component(Poly.treeStr1,values[:16],15,4,2.0,scalar)
    p2 = component(Poly.treeStr2,values[16:32],15,4,1.7,scalar)
    p3 = component(Poly.treeStr3,values[32:],27,8,2.0,scalar)
    one_half,two,half = scalar(1.5),scalar(2.0),scalar(0.5)
    def pool_relu(x):
        sign = p3(p2(p1(x)))
        for _ in range(4):
            sign = one_half*sign-two*sign*sign*sign
        return x*(sign+half)
    return pool_relu


def dense_pool(x,pool_relu):
    """Center-first comparisons; exclude invalid candidates (-inf padding)."""
    h,w = x.shape[-2:]
    current = x[:,:,::2,::2].clone()
    padded = F.pad(x,(1,1,1,1))
    yy,xx = torch.meshgrid(torch.arange(h//2),torch.arange(w//2),indexing='ij')
    for patch in range(9):
        if patch == 4:
            continue
        dy,dx = patch//3-1,patch%3-1
        other = padded[:,:,patch//3:patch//3+h:2,patch%3:patch%3+w:2]
        valid = (2*yy+dy>=0)&(2*yy+dy<h)&(2*xx+dx>=0)&(2*xx+dx<w)
        difference = other-current
        if float(difference[:,:,valid].abs().max()) > 1:
            raise ValueError('pool comparator outside normalized domain')
        current = current+pool_relu(difference)*valid
    return current


def dense(model,image,relu,pool_relu,exact_pool=False):
    def conv(x,layer,bn):
        gain = bn.weight/torch.sqrt(bn.running_var+bn.eps)
        bias = (bn.bias-bn.running_mean*gain)/BOUNDARY
        return F.conv2d(x,layer.weight,stride=layer.stride,padding=layer.padding)*gain[None,:,None,None]+bias[None,:,None,None]
    out = conv(image/BOUNDARY,model.conv1,model.bn1)
    stages = {'stem.conv':out}
    out = relu(out)
    stages['stem.relu'] = out
    out = F.max_pool2d(out,3,2,1) if exact_pool else dense_pool(out,pool_relu)
    stages['stem.maxpool'] = out
    for stage in range(1,5):
        for index,block in enumerate(getattr(model,f'layer{stage}')):
            shortcut = out
            if block.downsample is not None:
                shortcut = conv(out,block.downsample[0],block.downsample[1])
            out = relu(conv(out,block.conv1,block.bn1))
            out = relu(conv(out,block.conv2,block.bn2))
            out = relu(conv(out,block.conv3,block.bn3)+shortcut)
            stages[f'layer{stage}.{index}'] = out
    stages['head.pool'] = out.mean((2,3))
    return F.linear(stages['head.pool']*BOUNDARY,model.fc.weight,model.fc.bias)[0],stages


def pack_input(image):
    result = []
    for py in range(2):
        for px in range(2):
            phase = F.pad(image[0,:,py::2,px::2]/BOUNDARY,(0,16,0,16))
            result.extend(layout.pack(phase,layout.shape(3,128)))
    return result


def max_pool(values,source,pool_relu,clear):
    target = layout.shape(source['co'],source['ho']//2,source['ko']*2,source['nt'])
    src,dst = layout.indices(source),layout.indices(target)[:,:56,:56]
    c,y,x = np.indices(dst.shape)
    def candidate(dy,dx):
        yy,xx = y*2+dy,x*2+dx
        keep = (yy>=0)&(yy<112)&(xx>=0)&(xx<112)
        packs = layout.gather(values,src[c[keep],yy[keep],xx[keep]],dst[keep],
                              target['no'],source['nt'],clear)
        mask = np.zeros(target['no']*source['nt'])
        mask[dst[keep]] = 1
        return packs,mask.reshape(target['no'],source['nt'])
    current,_ = candidate(0,0)
    for patch in range(9):
        if patch == 4:
            continue
        other,valid = candidate(patch//3-1,patch%3-1)
        current = [a+pool_relu(b-a)*layout.plain(mask,clear)
                   for a,b,mask in zip(current,other,valid)]
    return current,target


def network(model,inputs,relu,pool_relu,observe=lambda *_:None,clear=False):
    def folded(layer,bn):
        gain = bn.weight.detach()/torch.sqrt(bn.running_var+bn.eps)
        return (layer.weight.detach()*gain[:,None,None,None]).numpy(),((bn.bias.detach()-bn.running_mean*gain)/BOUNDARY).numpy()
    def conv(values,source,layer,bn,valid):
        weight,bias = folded(layer,bn)
        return layout.conv(values,source,weight,bias,layer.stride[0],valid,clear)
    weight,bias = folded(model.conv1,model.bn1)
    stem = []
    for py in range(2):
        for px in range(2):
            kernel = np.zeros((64,3,5,5))
            for kh in range(7):
                for kw in range(7):
                    dy,dx = kh-3,kw-3
                    if dy%2==py and dx%2==px:
                        kernel[:,:,(dy-py)//2+2,(dx-px)//2+2] = weight[:,:,kh,kw]
            phase = py*2+px
            out,shape = layout.conv(inputs[phase*2:phase*2+2],layout.shape(3,128),kernel,
                                    bias if phase==0 else np.zeros(64),1,112,clear)
            stem.append(out)
    out = [layout.add(p[i] for p in stem) for i in range(shape['no'])]
    observe('stem.conv',out,shape)
    out = [relu(value) for value in out]
    observe('stem.relu',out,shape)
    out,shape = max_pool(out,shape,pool_relu,clear)
    observe('stem.maxpool',out,shape)
    valid = 56
    for stage in range(1,5):
        for index,block in enumerate(getattr(model,f'layer{stage}')):
            shortcut = out
            next_valid = valid//block.conv2.stride[0]
            if block.downsample is not None:
                shortcut,_ = conv(out,shape,block.downsample[0],block.downsample[1],next_valid)
            out,first = conv(out,shape,block.conv1,block.bn1,valid)
            out = [relu(v) for v in out]
            out,second = conv(out,first,block.conv2,block.bn2,next_valid)
            out = [relu(v) for v in out]
            out,shape = conv(out,second,block.conv3,block.bn3,next_valid)
            out = [relu(a+b) for a,b in zip(out,shortcut)]
            valid = next_valid
            observe(f'layer{stage}.{index}',out,shape)
    out = layout.global_pool(out,shape,valid,clear)
    observe('head.pool',out,layout.shape(2048,1))
    return layout.linear(out,model.fc.weight.detach().numpy()*BOUNDARY,
                         model.fc.bias.detach().numpy(),SLOTS,clear)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--relu-coefficients',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new output directory')
    import hecate as hc
    torch.set_num_threads(2)
    model = load_model(args.weights)
    constants = {}
    def scalar(value):
        if value not in constants:
            constants[value] = hc.Plain(np.full(SLOTS,value))
        return constants[value]
    @hc.func('c,c,c,c,c,c,c,c')
    def ResNet50Frontend(i0,i1,i2,i3,i4,i5,i6,i7):
        return network(model,[i0,i1,i2,i3,i4,i5,i6,i7],
                       make_relu(args.relu_coefficients,scalar),make_pool_relu(args.relu_coefficients,scalar))
    args.output.mkdir()
    print(hc.save(str(args.output),str(args.output),bytecode=True),flush=True)
    (args.output/'frontend.json').write_text(json.dumps(dict(model='ResNet50',slots=SLOTS,
        input_packs=8,boundary=BOUNDARY,pool='native polynomial maxpool with four sign refinements',
        manual_bootstraps=False,weights={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.weights.glob('*.txt')}),indent=2)+'\n')


if __name__ == '__main__':
    main()
