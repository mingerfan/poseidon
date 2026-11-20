"""Small independent clear references for the finite native cell profile.

Preparation consumes only an already saved training vector and public plan
metadata. No Torch, NumPy, key, evaluator, model, or ciphertext is constructed.
The per-instruction replay is checked against direct cell formulas before any
future native profiling. Bootstrap is a mathematical identity in both paths.
"""
import argparse
from array import array
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

from .r31_lattigo_binding import NativeBinding
from .r31_lattigo_profile import make_profile_plan
from .r31_lattigo_runtime_plan import content_filename, plan_inventory


def _vector(values, slots):
    result = array('d', values)
    if len(result) != slots or not all(math.isfinite(x) for x in result):
        raise ValueError('Expected a finite saved float64 packed vector')
    return result


def _operation(kind, a, b=None, attrs=None):
    attrs = attrs or {}
    if kind in ('add_cc', 'add_cp'):
        return array('d', (x+y for x,y in zip(a,b)))
    if kind in ('mul_cc', 'mul_cp'):
        return array('d', (x*y for x,y in zip(a,b)))
    if kind == 'rotate':
        step = attrs['steps'] % len(a)
        return a[step:]+a[:step]
    if kind == 'negate':
        return array('d', (-x for x in a))
    if kind in ('mod_switch', 'rescale', 'relinearize', 'boot'):
        return array('d', a)
    raise ValueError('Unsupported finite profile operation: '+str(kind))


def replay_profile(plan, constants, packed):
    """Replay generated instructions with last-use ownership; no CKKS semantics."""
    slots = len(packed)
    desc = {v['id']:v for v in plan['values']}
    encoded = {row['output']:row['payload']['content'] for row in plan['initialization']}
    last = {name:i for i,op in enumerate(plan['execution']) for name in op['inputs']}
    final = plan['final_outputs'][0]
    pool = {plan['external_inputs'][0]:_vector(packed, slots)}
    counts = Counter(); peak = len(pool)
    for index,op in enumerate(plan['execution']):
        inputs = [constants[encoded[name]] if desc[name]['kind']=='plaintext' else pool[name]
                  for name in op['inputs']]
        value = _operation(op['op'], *inputs, attrs=op.get('attrs'))
        value = _vector(value, slots)
        pool[op['output']] = value; counts[op['op']] += 1
        peak = max(peak,len(pool))
        for name in set(op['inputs']):
            if last[name] == index and name != final:
                pool.pop(name,None)
        if op['output'] not in last and op['output'] != final:
            pool.pop(op['output'],None)
    if set(pool) != {final}:
        raise ValueError('Unexpected terminal independent profile live set')
    return pool[final], dict(op_counts=dict(sorted(counts.items())),
        execution_completed=sum(counts.values()),peak_live_arrays=peak,
        peak_vector_payload_bytes=peak*slots*8,bootstrap_is_mathematical_identity=True,
        keygen_runs=0,encryption_runs=0,bootstrap_runs=0,decryption_runs=0)


def direct_profile_sum(metadata, packed, *, precision_bits):
    """Evaluate each metadata cell directly, independent of the setup instruction graph."""
    slots = len(packed); total = None
    def operand(meta):
        if meta['kind'] == 'plaintext':
            if meta['scale_log2'] != precision_bits or meta['components'] != 1:
                raise ValueError('Unsupported direct plaintext metadata')
            return array('d',[.125])*slots
        if (meta['scale_log2'],meta['components']) == (precision_bits,2):
            return array('d',packed)
        if (meta['scale_log2'],meta['components']) == (2*precision_bits,3):
            return array('d',(x*x for x in packed))
        if (meta['scale_log2'],meta['components']) == (2*precision_bits,2):
            return array('d',(.5*x for x in packed))
        raise ValueError('Unsupported direct ciphertext metadata')
    ordered = {row['index']:row for row in metadata['cells']}
    for target in metadata['target_operations']:
        cell = ordered[target['cell_index']]
        operands = [operand(x) for x in cell['inputs']]
        a = operands[0]; kind = cell['op']
        # Direct formulas deliberately use their own indexing/arithmetic path,
        # rather than the instruction interpreter's operation helper.
        if kind in ('add_cc','add_cp'):
            value = array('d',(a[i]+operands[1][i] for i in range(slots)))
        elif kind in ('mul_cc','mul_cp'):
            value = array('d',(a[i]*operands[1][i] for i in range(slots)))
        elif kind == 'rotate':
            step = cell['representative_attrs']['steps']
            value = array('d',(a[(i+step)%slots] for i in range(slots)))
        elif kind == 'negate':
            value = array('d',(-a[i] for i in range(slots)))
        elif kind in ('mod_switch','rescale','relinearize','boot'):
            value = array('d',a)
        else:
            raise ValueError('Unsupported direct profile cell: '+str(kind))
        # Relinearization, rescale and modulus drops that prepare aggregation
        # leave the clear mathematical value unchanged in this independent oracle.
        if total is None:
            total = value
        else:
            total = array('d',(x+y for x,y in zip(total,value)))
    if total is None:
        raise ValueError('No profile target cells')
    return _vector(total,slots)


def compare_references(reference, observed):
    if len(reference) != len(observed) or not reference:
        raise ValueError('Independent profile reference shapes differ')
    max_abs = max_ratio = 0.; passed = True
    for a,b in zip(reference,observed):
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('Nonfinite independent profile reference')
        delta = abs(a-b); allowed = 1e-12+1e-12*abs(a)
        max_abs=max(max_abs,delta);max_ratio=max(max_ratio,delta/allowed)
        passed = passed and delta<=allowed
    return dict(passed=passed,shape=[len(reference)],max_absolute_error=max_abs,
        max_error_ratio=max_ratio,criterion='1e-12+1e-12*abs(reference)')


def prepare(args):
    """Prepare a bounded profile plan/bundle and two-path float64 reference only."""
    began=time.perf_counter_ns()
    binding=NativeBinding(50,mod1_inv_degree=15,log_message_ratio=2,
                          bootstrap_normalization_log2=3)
    original=json.loads(args.plan.read_text())
    if len(original['execution'])!=17063 or sum(x['op']=='boot' for x in original['execution'])!=90:
        raise ValueError('The complete frozen source plan must contain17063 operations and90bootstraps')
    plan,manifest,metadata=make_profile_plan(original,binding=binding)
    if metadata['cell_count']!=36 or len(plan['execution'])!=126:
        raise ValueError('Expected all36 actual metadata cells with126 total profile operations')
    inventory=plan_inventory(plan,manifest,binding=binding)
    slots=inventory['slot_count']
    if args.input.stat().st_size!=slots*8:
        raise ValueError('Saved training packed input byte length differs')
    packed=array('d');packed.frombytes(args.input.read_bytes())
    if sys.byteorder!='little':packed.byteswap()
    packed=_vector(packed,slots)
    constants={row['content']:array('d',[row['value']])*slots for row in metadata['constant_vectors']}
    replay,summary=replay_profile(plan,constants,packed)
    direct=direct_profile_sum(metadata,packed,precision_bits=50)
    comparison=compare_references(direct,replay)
    if comparison['passed'] is not True:
        raise ValueError('Finite profile direct-cell and instruction references disagree')
    args.out.mkdir(parents=True,exist_ok=False)
    bundle=args.out/'profile.bundle';(bundle/'data').mkdir(parents=True)
    def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    write(args.out/'profile-plan.json',plan);write(bundle/'manifest.json',manifest)
    for content,value in constants.items():
        raw=array('d',value)
        if sys.byteorder!='little':raw.byteswap()
        (bundle/'data'/content_filename(content)).write_bytes(raw.tobytes())
    write(args.out/'expected.json',dict(indices=list(range(slots)),values=list(direct),
        scope='Independent36-cell clear reference; mathematical bootstrap identity; all16384 final slots compared',
        math_comparison=comparison))
    write(args.out/'profile-metadata.json',metadata)
    write(args.out/'reference-summary.json',summary)
    result=dict(status='prepared',mode='independent-native-profile-reference-preparation',
        source_plan=str(args.plan.resolve()),saved_train_input=str(args.input.resolve()),
        output_plan=str(args.out/'profile-plan.json'),output_bundle=str(bundle),
        expected=str(args.out/'expected.json'),metadata=str(args.out/'profile-metadata.json'),
        cell_count=36,execution_count=126,reference_comparison=comparison,
        native_executions=0,keygen_runs=0,encryption_runs=0,bootstrap_runs=0,decryption_runs=0,
        llm_calls=0,extra_integrity_checks=0,security_acceptance=None,formal_prediction=None,
        original_model_forward_runs=0,elapsed_ns=time.perf_counter_ns()-began,
        timing_scope='Clear plan/reference preparation only; native cell timing remains unmeasured')
    write(args.out/'preparation.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','input','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    result=prepare(parser.parse_args(argv))
    print(json.dumps(result,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
