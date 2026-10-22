"""The four R8a development models, using existing or separately trained weights."""

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .resnet20 import load_torch


REPO = Path(__file__).resolve().parents[5]
WEIGHTS_SILU = REPO / 'third_party/ckks-runtime/third_party/dacapo/examples/data/resnet20.silu.model'
NAMES = ('resnet20', 'resnet20_silu', 'cnn_plain', 'mlp')
IMAGENET_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CIFAR_NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
SILU_NORMALIZATION = IMAGENET_NORMALIZATION


@dataclass
class ZooModel:
    name: str
    module: torch.nn.Module
    example_input: torch.Tensor
    normalization: tuple

    def adapt(self, images: np.ndarray) -> np.ndarray:
        return images.reshape(len(images), 3072) if self.name == 'mlp' else images

    def import_graph(self):
        from ..intake import import_module
        return import_module(self.module, (self.example_input,), self.name, frontend="aten")


class CnnPlain(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc(x.mean(dim=(2, 3)))


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def get(name: str, models_dir: Path) -> ZooModel:
    if name not in NAMES:
        raise ValueError(f'Unknown development model {name}')
    normalization = IMAGENET_NORMALIZATION
    if name == 'resnet20':
        module = load_torch()
    elif name == 'resnet20_silu':
        source = REPO / 'third_party/ckks-runtime/third_party/dacapo/python/poly/poly/models/ResNet.py'
        spec = importlib.util.spec_from_file_location('pt_agent_dacapo_resnet', source)
        definition = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(definition)
        module = definition.resnet20()
        state = torch.load(WEIGHTS_SILU, map_location='cpu', weights_only=True)['state_dict']
        module.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()}, strict=True)
        normalization = SILU_NORMALIZATION
    else:
        module = CnnPlain() if name == 'cnn_plain' else Mlp()
        state = torch.load(Path(models_dir)/(name+'.pt'), map_location='cpu', weights_only=True)
        module.load_state_dict(state, strict=True)
    shape = (1, 3072) if name == 'mlp' else (1, 3, 32, 32)
    return ZooModel(name, module.float().eval(), torch.zeros(shape), normalization)
