#!/usr/bin/env python3
"""Summarize this experiment's real reports without mixing timing scopes."""
import argparse
import json
from pathlib import Path
import statistics


def native(path):
    data=json.loads(path.read_bytes())
    if not data.get('completed'): raise ValueError(f'incomplete run: {path}')
    rows=[r for r in data['runs'] if r['phase']=='measured']
    if not rows: raise ValueError(f'no measured iterations: {path}')
    values=[r['online_execution_ms'] for r in rows]
    outputs=data.get('outputs',[])
    result=dict(report=str(path),samples=len(rows),online_ms=values,
        median_online_ms=statistics.median(values),range_online_ms=[min(values),max(values)],
        median_runtime_wall_ms=statistics.median(r['wall_ms'] for r in rows),
        median_initialization_ms=statistics.median(r['initialization_ms'] for r in rows),
        median_runtime_setup_ms=statistics.median(r['setup_ms'] for r in rows),
        native_boot_calls_per_iteration=[r['native_boot_calls'] for r in rows],
        rotation_key_policy=data['rotation_key_policy'],allocator=data.get('allocator','managed'),
        fixture_sha256=data.get('fixture_sha256'),
        resident_plaintexts=data.get('resident_plaintexts'),device_measurements=data.get('device_measurements'),
        finite_output_slots=sum(o.get('finite_slots',0) for o in outputs),
        nonfinite_output_slots=sum(o.get('nonfinite_slots',0) for o in outputs),
        accuracy_validated=False,secret_key_assisted_refresh=data.get('secret_key_assisted_refresh'),
        context=data['context'],model=data['model'])
    return result


def require_shape(data, scope):
    model=data['model']; config=model['config']
    for key,value in dict(tokens=128,hidden=768,intermediate=3072,heads=12,layers=1).items():
        if config.get(key)!=value: raise ValueError(f'{scope}: incompatible {key}')
    if model.get('scope')!=scope: raise ValueError(f'expected {scope} report, not {model.get("scope")}')
    if model.get('seed')!=20261114: raise ValueError('report does not use the shared experiment seed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cerium-results',type=Path,required=True)
    p.add_argument('--native-encoder',type=Path)
    p.add_argument('--native-ffn-up',type=Path)
    p.add_argument('--native-ffn-up-unfused',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); cerium=json.loads(a.cerium_results.read_bytes())
    result=dict(format_version=1,cerium=cerium,
        comparison_contract=dict(same_gpu=True,tokens=128,hidden=768,intermediate=3072,encoder_blocks=1,
            same_dense_synthetic_weights_and_hidden_inputs=True,
            matched_he_parameters=False,matched_nonlinear_approximations=False,
            poseidon_scope='synchronized runtime online; excludes setup/init/input H2D/output D2H/decryption',
            cerium_scope='fullblock internal runtime; FFN-up Python program.run wall; not H2D+D2H wall',
            ratio_semantics='native systems with same model shape, not an equal-HE-parameter implementation speedup',
            accuracy_validated=False))
    if a.native_encoder:
        x=native(a.native_encoder); require_shape(x,'encoder'); result['poseidon_encoder']=x
        result['encoder_poseidon_over_cerium_runtime_ratio']=x['median_online_ms']/cerium['fullblock']['steady_runtime_internal_median_ms']
    if a.native_ffn_up:
        x=native(a.native_ffn_up); require_shape(x,'ffn-up'); result['poseidon_ffn_up']=x
        result['ffn_up_poseidon_over_cerium_run_wall_ratio']=x['median_online_ms']/cerium['ffn_up']['run_wall_median_ms']
    if a.native_ffn_up_unfused:
        x=native(a.native_ffn_up_unfused); require_shape(x,'ffn-up'); result['poseidon_ffn_up_unfused']=x
        if 'poseidon_ffn_up' in result:
            other=result['poseidon_ffn_up']
            for key in ('context','allocator','rotation_key_policy','fixture_sha256'):
                if x[key]!=other[key]: raise ValueError(f'Rescale ablation mismatches {key}')
            result['ffn_up_rescale_fusion_speedup']=x['median_online_ms']/result['poseidon_ffn_up']['median_online_ms']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k.endswith('_ratio') or k.endswith('_speedup')},indent=2))


if __name__=='__main__': main()
