"""Original DaCapo models selected for R13 common-model intake preparation."""

from ..models import dacapo_builders
from ..paths import DACAPO_UPSTREAM


_MODULE = 'pt_agent.experiments.r13_common_models'
FACTORIES = {
    _MODULE + ':resnet20_silu': {
        'name': 'resnet20_silu',
        'source': DACAPO_UPSTREAM / 'python/poly/poly/models/ResNet.py',
        'checkpoint': DACAPO_UPSTREAM / 'examples/data/resnet20.silu.model',
        'inference_preprocessing_source': DACAPO_UPSTREAM / 'examples/tests/ResNet.py',
        'spec_revision': 'R13',
    },
    _MODULE + ':vgg16_silu_avgpool': {
        'name': 'vgg16_silu_avgpool',
        'source': DACAPO_UPSTREAM / 'python/poly/poly/models/VGG16.py',
        'checkpoint': DACAPO_UPSTREAM / 'examples/data/vgg16_silu_avgpool_model',
        'inference_preprocessing_source': DACAPO_UPSTREAM / 'examples/tests/VGG16.py',
        'spec_revision': 'R13',
    },
    _MODULE + ':alexnet_silu_avgpool': {
        'name': 'alexnet_silu_avgpool',
        'source': DACAPO_UPSTREAM / 'python/poly/poly/models/AlexNet.py',
        'checkpoint': DACAPO_UPSTREAM / 'examples/data/alexNet_silu_avgpool_model',
        'inference_preprocessing_source': DACAPO_UPSTREAM / 'examples/tests/AlexNet.py',
        'spec_revision': 'R13',
    },
}


def resnet20_silu():
    import torch

    record = FACTORIES[_MODULE + ':resnet20_silu']
    definition = dacapo_builders._load_definition(record['source'], 'ResNet')
    model = torch.nn.DataParallel(definition.resnet20())
    checkpoint = torch.load(record['checkpoint'], map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.module.eval()


def vgg16_silu_avgpool():
    return dacapo_builders.vgg16_silu_avgpool()


def alexnet_silu_avgpool():
    return dacapo_builders.alexnet_silu_avgpool()
