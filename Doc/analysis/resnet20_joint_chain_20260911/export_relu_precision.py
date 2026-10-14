"""Export the saved Q50 witness and ORIGINAL application coefficients, no search.

The small text format is consumed by relu_precision.cpp. Leaf coefficients are
stored in heap-index order in d13.txt, not recursive traversal order.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_COEFFICIENTS = Path('/home/liufuyao/Work/poseidon_gpu_other/resnet20-9.3/benchmark/resnet20_gpu/data/resnet20/relu_param/d13.txt')


def leaves(tree, heap=1):
    if tree['type'] == 'leaf':
        return [(heap, tree)]
    return leaves(tree['remainder'], 2*heap) + leaves(tree['quotient'], 2*heap+1)


def attach_coefficients(stages, values):
    offset = 0
    for stage, factor in zip(stages, (0.5, 1/1.7, 0.5), strict=True):
        for _, node in sorted(leaves(stage['tree'])):
            count = node['degree'] + 1
            block = values[offset:offset+count]
            if len(block) != count:
                raise ValueError('truncated coefficient file')
            # The source evaluator reads odd positions only. d13.txt retains
            # numerical fit residues around 1e-26 at unused even positions.
            if any(abs(x) > 1e-20 for x in block[::2]):
                raise ValueError('unexpectedly large unused even coefficient')
            node['coefficients'] = [x*factor for x in block[1::2]]
            offset += count
    if offset != len(values):
        raise ValueError(f'unused coefficients: {len(values)-offset}')


def export(witness, coefficients):
    report = json.loads(witness.read_text())
    raw = coefficients.read_bytes()
    attach_coefficients(report['relu']['stages'], list(map(float, raw.split())))
    lines = []
    def emit(*values):
        lines.append(' '.join(format(x, '.17g') if isinstance(x, float) else str(x) for x in values))
    emit('RELU_PRECISION_V1')
    emit(len(report['q_bottom_first']), len(report['p']))
    emit(*report['q_bottom_first'], *report['p'])
    emit(len(report['relu']['stages']))
    def tree(node):
        emit(node['type'], node['work'], node['output'], node['pre'], node['scale'])
        if node['type'] == 'leaf':
            emit(node['degree'], *node['coefficients'])
        else:
            emit(node['split'])
            tree(node['quotient'])
            tree(node['remainder'])
    for stage in report['relu']['stages']:
        emit(stage['degree'], stage['start'], stage['end'], stage['input_scale'], stage['output_scale'], len(stage['basis']))
        for b in stage['basis']:
            emit(b['degree'], b['work'], b['output'], b['pre'], b['scale'])
        tree(stage['tree'])
    emit(report['relu']['tail_work'], report['relu']['tail_drop'])
    emit(hashlib.sha256(raw).hexdigest(), hashlib.sha256(witness.read_bytes()).hexdigest())
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--witness', type=Path, default=HERE/'q50.json')
    ap.add_argument('--coefficients', type=Path, default=DEFAULT_COEFFICIENTS)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.write_text(export(args.witness, args.coefficients))
