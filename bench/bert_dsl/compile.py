#!/usr/bin/env python3
"""Schedule an already physical BERT CKKS checkpoint without replanning Boot."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hecate-opt',type=Path,required=True)
    p.add_argument('--ckks-config',type=Path,required=True)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    source=a.input_dir.resolve(); out=a.output_dir.resolve()
    spec=source/'operator-spec.json'; checkpoint=source/'manual.ckks.mlir'
    for path in (a.hecate_opt,a.ckks_config,spec,checkpoint):
        if not path.is_file(): p.error(f'missing input: {path}')
    if out.exists(): p.error('--output-dir must be new')
    data=json.loads(spec.read_bytes()); profile=data['boot_profiles'][0]['profile_id']
    command=[str(a.hecate_opt.resolve()), '--ckks-to-runtime-plan',
        '--ckks-config='+str(a.ckks_config.resolve()), '--mlir-disable-threading',
        '--runtime-plan-device-counts=1', '--runtime-plan-operator-spec-path='+str(spec),
        '--runtime-plan-operator-spec-id='+data['spec_id'],
        '--runtime-plan-operator-spec-sha256=sha256:'+hashlib.sha256(spec.read_bytes()).hexdigest(),
        '--runtime-plan-context-id='+data['context']['context_id'],
        '--runtime-plan-target-id='+data['target_id'],
        '--runtime-plan-boot-profile='+profile, '--runtime-plan-boot-implementation=native',
        '--runtime-plan-memory', '--runtime-plan-memory-reuse=false',
        '--runtime-plan-coalesce-rescales=false', '--runtime-plan-levels-per-logical-level=1',
        '--runtime-plan-format=json', '--runtime-plan-inline-payload-max-bytes=0',
        '--runtime-plan-bundle-format=pack', '--emit-bytecode', str(checkpoint),
        '-o', str(out/'compiled.mlirbc')]
    print(json.dumps(command,indent=2),flush=True)
    if a.dry_run: return
    out.mkdir(parents=True)
    (out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    with (out/'compile.log').open('wb') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    plans=list(out.glob('*.runtime-plan.json'))
    if len(plans)!=1: raise RuntimeError('expected one exported plan')
    plan=json.loads(plans[0].read_bytes())
    boots=[row for phase in ('initialization','execution','finalization') for row in plan[phase]
           if row.get('op')=='boot']
    assert all(row['attrs']['implementation']=='native' for row in boots)
    print(json.dumps(dict(plan=str(plans[0]),native_boots=len(boots),ckks_executed=False)))


if __name__=='__main__': main()
