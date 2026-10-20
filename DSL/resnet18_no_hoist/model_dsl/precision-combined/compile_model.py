"""Add a model name to the existing artifact CLI; use full stock Dacapo planning."""
import importlib.util
import sys
from pathlib import Path

if __name__ == '__main__':
    frame = Path(__file__).resolve().parent.parent/'resnet20-framework-v1'
    path = frame/'poseidon/third_party/ckks-runtime/integrations/dacapo/generate_model_artifacts.py'
    spec = importlib.util.spec_from_file_location('artifact_cli',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODELS['resnet18'] = 'ResNet18Frontend'
    if '--defer-plaintext' in sys.argv:
        sys.argv.remove('--defer-plaintext')
        original = module.build_optimizer_command
        def deferred(*args, **kwargs):
            command = original(*args, **kwargs)
            command.remove('--enable-debug-printer')
            command += ['--runtime-plan-defer-plaintext',
                        '--mlir-elide-elementsattrs-if-larger=16']
            return command
        module.build_optimizer_command = deferred
    module.main()
