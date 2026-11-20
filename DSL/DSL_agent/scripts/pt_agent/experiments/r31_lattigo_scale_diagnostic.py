"""Clear arithmetic diagnosis of native scale alignment, never a golden oracle.

The actual prime values come from the completed native constructor. Addition
uses Lattigo's integer-truncated ratio and max-scale output. Native bootstrap
is idealized as mathematical identity with default output scale; noise,
encoding rounding and bootstrap approximation errors remain unmodeled.
"""
from fractions import Fraction
import argparse
import json
import math
import os
from pathlib import Path
import time

from .r31_lattigo_runtime_plan import content_filename,plan_inventory
from .r10_paper_metrics import write_json


def addition_factors(left_scale,right_scale):
    target=max(left_scale,right_scale)
    left=left_scale*(target//left_scale)/target
    right=right_scale*(target//right_scale)/target
    return left,right,target


def bootstrap_normalization_factor(scale,q0,*,log_message_ratio=8,evalmod_log_scale=60):
    """Coefficient/metadata ratio introduced by native ScaleDown and ModUp.

    This case uses a level-one input that ScaleDown first drops to zero. The
    integer scalar multipliers affect coefficients; ModUp's unrounded ratio
    updates the metadata. Nonlinear modular approximation is not simulated.
    """
    def nearest(x):return (2*x.numerator+x.denominator)//(2*x.denominator)
    first=nearest(Fraction(q0)/(scale*2**log_message_ratio))
    if first<1:raise ValueError('ScaleDown would lack a positive multiplier')
    target=Fraction(2**(evalmod_log_scale-log_message_ratio))
    ratio=target/(scale*first)
    second=nearest(ratio) if ratio>1 else 1
    return scale*first*second/target,first,second


def replay_scale(plan,manifest,bundle,packed,actual_q,*,bootstrap_rows=None,boot_normalization=False,
                 log_message_ratio=8,evalmod_log_scale=60):
    import numpy as np
    inventory=plan_inventory(plan,manifest);slots=inventory['slot_count']
    values={v['id']:v for v in plan['values']}
    encodes={e['output']:e['payload']['content'] for e in plan['initialization']}
    scales={name:Fraction(2**values[name]['scale_log2']) for name in encodes}
    first=plan['external_inputs'][0];scales[first]=Fraction(2**values[first]['scale_log2'])
    source=np.asarray(packed)
    if source.dtype!=np.float64 or source.shape!=(slots,) or not np.isfinite(source).all():
        raise ValueError('Finite original packed float64 values are required')
    pool={first:source.copy()};last={name:i for i,op in enumerate(plan['execution']) for name in op['inputs']}
    final=plan['final_outputs'][0];cache_name=None;cache=None;changes=[];observed=[]
    native_rows={row['ordinal']:row for row in bootstrap_rows or ()}
    def plain(name):
        nonlocal cache_name,cache
        if name!=cache_name:
            path=bundle/'data'/content_filename(encodes[name])
            if path.stat().st_size!=slots*8:raise ValueError('Original constant size differs')
            cache=np.fromfile(path,dtype='<f8',count=slots);cache_name=name
            if not np.isfinite(cache).all():raise ValueError('Nonfinite original constants')
        return cache
    for index,op in enumerate(plan['execution']):
        names=op['inputs'];kind=op['op'];a=pool[names[0]];sa=scales[names[0]]
        if len(names)==2:
            b=plain(names[1]) if values[names[1]]['kind']=='plaintext' else pool[names[1]];sb=scales[names[1]]
        with np.errstate(over='raise',invalid='raise'):
            if kind in ('add_cc','add_cp'):
                fa,fb,scale=addition_factors(sa,sb)
                out=a*float(fa)+b*float(fb)
                if fa!=1 or fb!=1:
                    changes.append(dict(ordinal=op['ordinal'],op=kind,left_factor=float(fa),right_factor=float(fb)))
            elif kind in ('mul_cc','mul_cp'):out=a*b;scale=sa*sb
            elif kind=='rescale':out=a.copy();scale=sa/actual_q[values[names[0]]['level']]
            elif kind=='rotate':out=np.roll(a,-op['attrs']['steps']);scale=sa
            elif kind=='negate':out=-a;scale=sa
            elif kind in ('mod_switch','relinearize'):out=a.copy();scale=sa
            elif kind=='boot':
                predicted=math.log2(sa.numerator)-math.log2(sa.denominator)
                row=dict(ordinal=op['ordinal'],predicted_entry_scale_log2=predicted,
                         actual_entry_scale_log2=native_rows.get(op['ordinal'],{}).get('input_scale_log2'))
                if boot_normalization:
                    if values[names[0]]['level']!=1:raise ValueError('Normalization diagnosis is limited to the observed level-one boot inputs')
                    factor,first,second=bootstrap_normalization_factor(sa,actual_q[0],
                        log_message_ratio=log_message_ratio,evalmod_log_scale=evalmod_log_scale)
                    row.update(normalization_factor=float(factor),scale_down_integer=first,mod_up_integer=second)
                    out=a*float(factor)
                else:out=a.copy()
                observed.append(row);scale=Fraction(2**values[op['output']]['scale_log2'])
            else:raise ValueError('Unsupported scale diagnostic operation')
        if not np.isfinite(out).all():raise ValueError('Nonfinite scale diagnosis')
        pool[op['output']]=out;scales[op['output']]=scale
        for name in set(names):
            if last[name]==index and name!=final:pool.pop(name,None)
        if op['output'] not in last and op['output']!=final:pool.pop(op['output'],None)
    deltas=[abs(row['predicted_entry_scale_log2']-row['actual_entry_scale_log2']) for row in observed
            if row['actual_entry_scale_log2'] is not None]
    return pool[final],dict(mode='native-scale-alignment-clear-diagnostic',
        golden_reference_replaced=False,actual_ckks_executions=0,
        bootstrap_mode=('Native ScaleDown/ModUp integer normalization; nonlinear approximation/noise omitted' if boot_normalization else 'Mathematical identity; default output scale'),
        addition_rule='Truncate positive ratio to integer, scale lower operand coefficients by that integer, output=max operand scale',
        affected_additions=len(changes),addition_rows=changes,bootstrap_scale_rows=observed,
        max_bootstrap_entry_scale_log2_difference=max(deltas,default=None),
        actual_residual_q=actual_q,unmodeled=['native encoding rounding','native encryption/key-switch noise','native rescale coefficient rounding','bootstrap approximation and normalization effects'],
        formal_acceptance=None,accuracy_acceptance=None)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','bundle','input','native-result','golden','work-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--boot-normalization',choices=('identity','native-rounded'),default='identity')
    args=p.parse_args(argv)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    args.work_dir.mkdir(parents=True,exist_ok=False);began=time.perf_counter_ns()
    result=dict(round='R31',mode='native-scale-alignment-clear-diagnostic',status='running',error=None,
        llm_calls=0,keygen_runs=0,real_ckks_runs=0,intermediate_decryptions=0,formal_acceptance=None,
        golden_reference_replaced=False,accuracy_acceptance=None)
    try:
        import numpy as np
        native=json.loads(args.native_result.read_text());plan=json.loads(args.plan.read_text())
        manifest=json.loads((args.bundle/'manifest.json').read_text());golden=json.loads(args.golden.read_text())
        if native.get('compute_completed')!=len(plan['execution']) or native.get('final_decryption_completed')!=1:
            raise ValueError('Use the completed original full native diagnostic')
        packed=np.fromfile(args.input,dtype='<f8',count=16384)
        output,diagnostic=replay_scale(plan,manifest,args.bundle,packed,native['residual_parameters']['q'],
            bootstrap_rows=native['bootstrap_observations'],boot_normalization=args.boot_normalization=='native-rounded',
            log_message_ratio=native['bootstrap_parameters']['Mod1ParametersLiteral']['LogMessageRatio'],
            evalmod_log_scale=native['bootstrap_parameters']['Mod1ParametersLiteral']['LogScale'])
        decoded=output[golden['indices']];expected=np.asarray(golden['values'],dtype=np.float64)
        observed=np.asarray(native['final_selected_values'],dtype=np.float64)
        total=observed-expected;explained=decoded-expected;remaining=observed-decoded
        result.update(diagnostic=diagnostic,source=dict(plan=str(args.plan),bundle=str(args.bundle),input=str(args.input),
            native_result=str(args.native_result),golden=str(args.golden)),scale_diagnostic_values=decoded.tolist(),
            golden_values=expected.tolist(),actual_native_values=observed.tolist(),
            scale_diagnostic_minus_golden=explained.tolist(),native_minus_scale_diagnostic=remaining.tolist(),
            max_native_minus_golden=float(np.abs(total).max()),
            max_scale_diagnostic_minus_golden=float(np.abs(explained).max()),
            max_native_minus_scale_diagnostic=float(np.abs(remaining).max()),
            native_minus_golden_l2=float(np.linalg.norm(total)),native_minus_scale_diagnostic_l2=float(np.linalg.norm(remaining)),
            scope='Noise-free scale-alignment diagnosis only. Bootstrap remains ideal identity. Does not replace the original 1e-12 golden reference or original failed CKKS judgment.')
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    result['elapsed_ns']=time.perf_counter_ns()-began;write_json(args.work_dir/'scale-result.json',result)
    print(json.dumps({k:result.get(k) for k in ('status','error','max_native_minus_golden','max_scale_diagnostic_minus_golden','max_native_minus_scale_diagnostic')}))
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
