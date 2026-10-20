"""Server development gate: independent clear semantics, then Earth bytecode."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRAME = ROOT.parent / 'resnet20-framework-v1'
DATA = ROOT.parent / 'resnet18-model-dev/package-model/resnet18'
PYTHON = '/opt/conda/envs/apollo/bin/python'
env = dict(os.environ, HECATE=str(FRAME/'dacapo'), HECATE_BUILD=str(FRAME/'build-make'),
           PYTHONPATH=f"{FRAME}/dacapo/python/hecate:{ROOT.parent}/resnet18-model-dev/poly")


def run(command, log):
    with (ROOT/log).open('x') as output:
        subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                       stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    status = ROOT/'frontend.status'
    try:
        status.write_text('CLEAR_SEMANTICS_RUNNING\n')
        weights = str(DATA/'pretrained_parameters/resnet18_imagenet')
        d13 = '/home/hanshubin/Trident/resnet18/relu_param/d13.txt'
        run([PYTHON, '-u', 'check_semantics.py', '--weights', weights,
             '--relu-coefficients', d13, '--values', str(DATA/'testFile/test_values.txt'),
             '--labels', str(DATA/'testFile/test_label.txt'), '--native-logits',
             str(ROOT.parent/'resnet18-model-dev/native-logits.json'),
             '--report', 'semantics-final.json', '--runtime-fixture', 'fixture-final.json'],
            'semantics-final.log')
        status.write_text('EARTH_TRACE_RUNNING\n')
        run([PYTHON, '-u', 'ResNet18.py', '--weights', weights,
             '--relu-coefficients', d13, '--output', str(ROOT/'traced'), '--bytecode'],
            'trace.log')
        status.write_text('SEMANTICS_AND_EARTH_TRACE_PASS\n')
    except Exception:
        status.write_text('FAILED; inspect first error\n')
        raise
