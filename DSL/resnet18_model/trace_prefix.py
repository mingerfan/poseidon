"""Trace the real stem through d13 ReLU as an independent encrypted prefix gate."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import hecate as hc
from ResNet18 import BOUNDARY, SLOTS, load_model, load_image, dense, make_relu, network
import imagenet_layout as layout

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent/'resnet18-model-dev/package-model/resnet18'
D13 = Path('/home/hanshubin/Trident/resnet18/relu_param/d13.txt')


class PrefixComplete(Exception):
    def __init__(self, values):
        self.values = values


def observe(name, values, shape):
    if name == 'stem.relu':
        raise PrefixComplete(values)


if __name__ == '__main__':
    torch.set_num_threads(2)
    model = load_model(DATA/'pretrained_parameters/resnet18_imagenet')
    image = load_image(DATA/'testFile/test_values.txt')
    with torch.no_grad():
        _, stages = dense(model,image,make_relu(D13,float))
    expected = F.pad(stages['stem.relu'][0],(0,16,0,16))
    expected = layout.pack(expected,layout.shape(64,128))[0][:1000].tolist()
    constants = {}

    def scalar(value):
        if value not in constants:
            constants[value] = hc.Plain(np.full(SLOTS,value))
        return constants[value]

    @hc.func('c,c,c,c,c,c,c,c')
    def ResNet18Stem(i0,i1,i2,i3,i4,i5,i6,i7):
        try:
            network(model,[i0,i1,i2,i3,i4,i5,i6,i7],make_relu(D13,scalar),observe)
        except PrefixComplete as prefix:
            return prefix.values[0]
        raise AssertionError('stem prefix did not stop')

    out = ROOT/'traced-prefix'
    out.mkdir()
    print(hc.save(str(out),str(out),bytecode=True),flush=True)
    fixture = json.loads((ROOT/'fixture-final.json').read_text())
    fixture.update(python_output=expected,scope='stem convolution and d13 ReLU prefix, first pack',
                   model_sha256=hashlib.sha256((fixture['model_sha256']+':stem.relu').encode()).hexdigest())
    (ROOT/'fixture-prefix.json').write_text(json.dumps(fixture,allow_nan=False)+'\n')
