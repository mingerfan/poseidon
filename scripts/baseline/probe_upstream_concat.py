"""Run actual upstream packing/HE_Concat on plaintext slot vectors, never FHE.

Only named functions from trusted repository sources are compiled. No generated
Agent code, model checkpoint, API credential, or network is involved.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shlex
import tempfile

ROOT = Path(__file__).resolve().parents[2]
MPCB = ROOT / 'third_party/dacapo/python/poly/poly/MPCB.py'
FUNC = MPCB.with_name('Func.py')


class PackingTransforms:
    """Explicit diagnostic substitute for five einops packing patterns only.

    The environment does not have einops. This is not a production frontend or
    evidence that the real einops dependency has executed successfully.
    """
    @staticmethod
    def repeat(value, pattern, **sizes):
        if pattern == 'a -> (po a)':
            return value.repeat(sizes['po'])
        if pattern == '(ni a) -> ni (pi a)':
            return value.reshape(sizes['ni'], -1).repeat(1, sizes['pi'])
        raise ValueError('Unknown diagnostic repeat pattern: '+pattern)

    @staticmethod
    def rearrange(value, pattern, **sizes):
        if pattern in ('(ti s1 s2) h w -> (ti h s1 w s2)',
                       '(to s1 s2) h w -> (to h s1 w s2)'):
            c,h,w = value.shape
            s1,s2 = sizes['s1'],sizes['s2']
            return value.reshape(c//(s1*s2),s1,s2,h,w).permute(0,3,1,4,2).reshape(-1)
        if pattern == '(no a) -> no a':
            return value.reshape(sizes['no'], -1)
        raise ValueError('Unknown diagnostic rearrange pattern: '+pattern)


def upstream(mpcb_source=None):
    import numpy as np
    import torch
    space = dict(np=np, torch=torch, F=torch.nn.functional, einops=PackingTransforms,
                 Empty=lambda: None)
    for path, names in [(MPCB, {'cint', 'fint', 'roll', 'shapeClosure',
                                'InferShapes', 'CascadeConcat'}),
                        (FUNC, {'HE_Concat'})]:
        tree = ast.parse(mpcb_source if path == MPCB and mpcb_source is not None else path.read_text())
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in selected} != names:
            raise ValueError('Upstream function inventory changed')
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), space)
    return space


class PlainSlots:
    """Diagnostic slot arithmetic; explicitly not an encryption implementation."""
    def __init__(self, values):
        import numpy as np
        self.values = np.asarray(values, dtype=np.float64)

    def rotate(self, step):
        import numpy as np
        # Verified Hecate positive rotation is a left shift. MPCB.roll negates it.
        return PlainSlots(np.roll(self.values, -step))

    def __mul__(self, other):
        return PlainSlots(self.values * (other.values if isinstance(other, PlainSlots)
                                        else other.detach().numpy()))

    def __add__(self, other):
        return PlainSlots(self.values + other.values)


def packed_reference(tensor, *, nt, k, bb):
    """Independent scalar-index oracle; no einops or upstream packing calls."""
    import numpy as np
    _, channels, height, width = tensor.shape
    tiles = (channels + k*k - 1) // (k*k)
    size = tiles * height * k * width * k
    n = (size + nt - 1) // nt
    p = 1
    while size * p * 2 <= nt:
        p *= 2
    base = np.zeros(n * nt // p, dtype=np.float64)
    for c in range(channels):
        tile, rem = divmod(c, k*k)
        subrow, subcol = divmod(rem, k)
        for y in range(height):
            for x in range(width):
                index = (((tile * height + y) * k + subrow) * width + x) * k + subcol
                base[index] = float(tensor[0, c, y, x]) / bb
    result = np.empty((n, nt), dtype=np.float64)
    for row in range(n):
        for slot in range(nt):
            result[row, slot] = base[row * (nt // p) + slot % (nt // p)]
    return result


def shapes(api, nt, c, h, w, k, bb):
    return api['InferShapes'](dict(nt=nt, bb=bb, fh=1, fw=1, s=1,
                                  hi=h, wi=w, ki=k, ci=c, co=c))


def run_case(api, *, name, nt, c, h, w, k=1, bb=2):
    import numpy as np
    import torch
    previous = shapes(api, nt, c, h, w, k, bb)
    joined = api['CascadeConcat'](previous, previous)
    left = torch.arange(1, c*h*w+1, dtype=torch.float64).reshape(1,c,h,w) / 8
    right = -left - 3
    branch = api['shapeClosure'](**previous)
    close = api['shapeClosure'](**joined)
    left_packed, right_packed = branch['OP'](left), branch['OP'](right)
    for tensor, packed in [(left, left_packed), (right, right_packed)]:
        np.testing.assert_array_equal(packed.numpy(), packed_reference(tensor, nt=nt, k=k, bb=bb))
        np.testing.assert_array_equal(close['MPP'](tensor).numpy(), packed.numpy())
    logical = torch.cat((left, right), dim=1)
    expected = packed_reference(logical, nt=nt, k=k, bb=bb)
    np.testing.assert_array_equal(close['OP'](logical).numpy(), expected)
    def objects(values):
        out = np.empty(len(values), dtype=object)
        for i, row in enumerate(values):
            out[i] = PlainSlots(row.numpy())
        return out
    actual = api['HE_Concat'](close, objects(left_packed), objects(right_packed))
    got = np.stack([v.values for v in actual])
    delta = np.abs(got - expected)
    mismatch = np.argwhere(delta != 0)
    return dict(name=name, parameters=dict(nt=nt,c=c,h=h,w=w,k=k,bb=bb),
                shapes=joined, path='aligned' if c*h*w % nt == 0 else 'partial',
                passed=bool(np.array_equal(got,expected)), mismatched_slots=len(mismatch),
                max_absolute_error=float(delta.max()),
                first_mismatches=[dict(cipher=int(i),slot=int(j),actual=float(got[i,j]),
                                      reference=float(expected[i,j])) for i,j in mismatch[:8]])


CASES = [
    dict(name='replicated_partial',nt=16,c=1,h=1,w=3),
    dict(name='single_partial',nt=16,c=1,h=2,w=3),
    dict(name='half_cipher',nt=16,c=2,h=2,w=2),
    dict(name='aligned',nt=16,c=4,h=2,w=2),
    dict(name='aligned_multi',nt=16,c=8,h=2,w=2),
    dict(name='crosses_output_boundary',nt=16,c=2,h=1,w=5),
    dict(name='multi_partial_half',nt=16,c=3,h=2,w=4),
    dict(name='multi_partial_small_tail',nt=16,c=5,h=1,w=4),
    dict(name='interleaved_single',nt=32,c=4,h=1,w=3,k=2),
    dict(name='interleaved_multi',nt=32,c=8,h=2,w=3,k=2),
]


def audit():
    api = upstream()
    rows = [run_case(api, **case) for case in CASES]
    first = shapes(api,16,2,1,2,1,2)
    second = shapes(api,16,2,1,3,1,2)
    try:
        accepts_mismatch = isinstance(api['CascadeConcat'](first,second), dict)
    except ValueError:
        accepts_mismatch = False
    return dict(schema=1, mode='plaintext_upstream_layout_diagnostic',
                encrypted_execution=False, agent_calls=0,
                einops_execution=False, packing_transform_adapter='five_explicit_patterns', source_sha256={
                    str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (MPCB,FUNC)}, cases=rows,
                passed=sum(r['passed'] for r in rows), failed=sum(not r['passed'] for r in rows),
                mismatched_spatial_shapes_accepted=accepts_mismatch,
                limitation='Actual trusted helper source with explicit tensor reshape/repeat adapter; not installed einops, real FHE or generated Agent compatibility.')


def main():
    from hecate_python_env import VENV, enter_nix, WORK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Wrong workspace')
    if not args.inside:
        command = 'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 ' + shlex.join(
            [str(VENV/'bin/python'),str(Path(__file__).resolve()),'--inside'])
        return enter_nix(command,seconds=180)
    report = audit()
    result = Path(tempfile.mkdtemp(prefix='upstream-concat-layout-',dir=WORK/'results'))
    (result/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for row in report['cases']:
        print(row['name'], 'PASS' if row['passed'] else 'MISMATCH', row['max_absolute_error'])
    print('Mismatched spatial shapes accepted:', report['mismatched_spatial_shapes_accepted'])
    print('Plaintext diagnostic only. Evidence:',result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
