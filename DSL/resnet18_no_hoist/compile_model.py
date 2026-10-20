"""Register ResNet18 with the existing artifact CLI and enable deferred plaintexts."""
import argparse
import importlib.util
import sys
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--runtime-root', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    path = args.runtime_root/'integrations/dacapo/generate_model_artifacts.py'
    spec = importlib.util.spec_from_file_location('artifact_cli',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODELS['resnet18'] = 'ResNet18Frontend'
    original = module.build_optimizer_command
    def deferred(*args, **kwargs):
        command = original(*args, **kwargs)
        command.remove('--enable-debug-printer')
        return command+['--runtime-plan-defer-plaintext','--mlir-elide-elementsattrs-if-larger=16']
    module.build_optimizer_command = deferred
    sys.argv = [sys.argv[0],*remaining]
    module.main()
