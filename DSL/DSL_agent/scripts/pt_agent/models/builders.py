"""Development fixtures: module construction and weights, with no input policy."""

import importlib.util
from pathlib import Path

from ..paths import DACAPO_ROOT, RESULTS_DIR


def resnet20():
    from .resnet20 import load_torch
    return load_torch()


def resnet20_silu():
    import torch
    source = DACAPO_ROOT / 'python/poly/poly/models/ResNet.py'
    spec = importlib.util.spec_from_file_location('pt_agent_dacapo_resnet', source)
    definition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(definition)
    module = definition.resnet20()
    state = torch.load(DACAPO_ROOT / 'examples/data/resnet20.silu.model',
                       map_location='cpu', weights_only=True)['state_dict']
    module.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()}, strict=True)
    return module.float().eval()


def _small(name, models_dir=None):
    import torch
    from .zoo import CnnPlain, Mlp
    module = {'cnn_plain': CnnPlain, 'mlp': Mlp}[name]()
    state = torch.load(Path(models_dir or RESULTS_DIR / 'models-v2') / (name+'.pt'),
                       map_location='cpu', weights_only=True)
    module.load_state_dict(state, strict=True)
    return module.float().eval()


def cnn_plain():
    return _small('cnn_plain')


def mlp():
    return _small('mlp')


def load(name, models_dir=None):
    """Internal compatibility mapping for old experiment names only."""
    if name in ('cnn_plain', 'mlp'):
        return _small(name, models_dir)
    return {'resnet20': resnet20, 'resnet20_silu': resnet20_silu}[name]()
