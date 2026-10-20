"""Restore the original data archive and verify all 102 weight byte hashes."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir()
    with tarfile.open(ROOT/'fixtures/model_data.tar.gz') as archive:
        for entry in archive:
            target = (output/entry.name).resolve()
            assert target.is_relative_to(output) and entry.isfile(),entry.name
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(entry) as source:
                target.write_bytes(source.read())
    expected = json.loads((ROOT/'fixtures/model-source-hashes.json').read_text())
    weights = output/'resnet18/pretrained_parameters/resnet18_imagenet'
    actual = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in weights.glob('*.txt')}
    assert actual == expected['weights']
    assert hashlib.sha256((ROOT/'fixtures/d13.txt').read_bytes()).hexdigest() == expected['relu_sha256']
    print('PASS: 102 original weights and d13; data:',output/'resnet18')
