#!/usr/bin/env python3
"""Compile the full pinned trace with unchanged HEFT for one or all four GPUs."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--devices',type=int,nargs='+',choices=[1,4],default=[1,4])
    p.add_argument('--source',type=Path)
    p.add_argument('--output-name',default='qwen3-plans-bsgs-q50')
    p.add_argument('--gpu-budget-bytes',type=int,default=31*2**30)
    p.add_argument('--reserve-bytes',type=int,default=6*2**30)
    p.add_argument('--workspace-bytes',type=int,default=1024*2**20)
    p.add_argument('--lookahead-groups',type=int,default=128)
    p.add_argument('--prepare-backend',choices=['cpu','gpu','auto'],default='auto')
    p.add_argument('--boot-placement',choices=['optimized','greedy','depth-dp'],default='depth-dp')
    p.add_argument('--boot-profile',default='poseidon-gpu-degree58-da2-scale45',
                   help='Native degree58/DA2 GPU Boot, output level22/scale45')
    a=p.parse_args()
    if a.lookahead_groups<=0:p.error('--lookahead-groups must be positive')
    base=a.workspace.resolve();tool=base/'toolchain';profiles=base/'profiles'
    source=a.source.resolve() if a.source else base/'qwen3-traced-bsgs/trace_qwen3.mlirbc'
    spec_path=profiles/'operator-spec.json';spec=json.loads(spec_path.read_text())
    matches=[b for b in spec['boot_profiles'] if b['profile_id']==a.boot_profile]
    if len(matches)!=1:p.error('--boot-profile must select exactly one OperatorSpec profile')
    boot=matches[0]
    scale=spec['context']['default_scale_log2']
    compiler_profile=json.loads((profiles/'compiler-profile.json').read_text())
    compiler_profile['polynomialDegree']=spec['context']['poly_degree']
    bits=spec['context']['rns_moduli_log2']
    if not bits or len(set(bits))!=1:p.error('threshold Rescale currently requires equal-width Q primes')
    if spec['rescale_mode']!='native' or len(spec['context'].get('rns_moduli',[]))!=len(bits):
        p.error('threshold Rescale requires native mode and the actual Q chain')
    if boot['output_scale_log2']<scale:p.error('Boot output scale must be at least the initial scale')
    budget=boot['output_level']-spec['levels']['lower_bound']
    if budget<1:p.error('Boot has no usable Q budget')
    compiler_profile.update(rescalingFactor=bits[0],levelUpperBound=budget+1,
                            bootstrapLevelUpperBound=budget+1,bootstrapLevelLowerBound=1)
    digest='sha256:'+hashlib.sha256(spec_path.read_bytes()).hexdigest()
    prefix=[str(tool/'lib/ld-linux-x86-64.so.2'),'--library-path',
            f'{tool}/lib:/opt/conda/lib:/lib/x86_64-linux-gnu',str(tool/'bin/hecate-opt')]
    env=os.environ.copy();env.update(HECATE=str(tool),HECATE_BUILD=str(tool),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='12')
    failed=False
    for count in a.devices:
        out=base/a.output_name/f'gpu{count}';out.mkdir(parents=True,exist_ok=False)
        compiler_path=out/'compiler-profile.json'
        compiler_path.write_text(json.dumps(compiler_profile,indent=2)+'\n')
        command=prefix+['--dacapo',f'--boot-placement={a.boot_placement}',f'--ckks-config={compiler_path}',
          f'--waterline={scale}','--mlir-timing','--mlir-print-op-on-diagnostic=false',
          '--mlir-elide-elementsattrs-if-larger=16',f'--runtime-plan-id={3100+count}',
          f'--runtime-plan-target-id={spec["target_id"]}','--runtime-plan-capability-version=1',
          f'--runtime-plan-operator-spec-id={spec["spec_id"]}',f'--runtime-plan-operator-spec-version={spec["version"]}',
          f'--runtime-plan-operator-spec-sha256={digest}',f'--runtime-plan-context-id={spec["context"]["context_id"]}',
          '--runtime-plan-ntt=true',f'--runtime-plan-boot-profile={boot["profile_id"]}',
          f'--runtime-plan-boot-implementation={boot["implementation"]}','--runtime-plan-inline-payload-max-bytes=4096',
          f'--runtime-plan-physical-level-operator-spec-path={spec_path}','--runtime-plan-levels-per-logical-level=1',
          f'--runtime-plan-device-count={count}',f'--runtime-plan-device-counts={count}',
          f'--runtime-plan-operator-spec-path={spec_path}',f'--runtime-plan-communication-profile-path={profiles}/communication-profile.json',
          '--runtime-plan-memory=true','--runtime-plan-memory-reuse=true','--runtime-plan-memory-report=true',
          '--runtime-plan-plaintext-schedule=waterline',f'--runtime-plan-prepare-backend={a.prepare_backend}',
          f'--runtime-plan-gpu-budget-bytes={a.gpu_budget_bytes}',f'--runtime-plan-reserve-bytes={a.reserve_bytes}',
          f'--runtime-plan-workspace-bytes-per-op={a.workspace_bytes}',
          '--runtime-plan-host-budget-bytes=68719476736','--runtime-plan-pinned-budget-bytes=1073741824',
          '--runtime-plan-raw-budget-bytes=67108864','--runtime-plan-waterline-fraction=0.5',
          f'--runtime-plan-lookahead-groups={a.lookahead_groups}','--runtime-plan-format=binary','--runtime-plan-bundle-format=pack',
          '--runtime-plan-report-io=true',str(source),'-o',str(out/'qwen3.mlir')]
        command.append(f'--bootstrap-scale={boot["output_scale_log2"]}')
        (out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        # Each full-model compile can exceed 130 GiB of Host memory. Finish one
        # device configuration before adding another full-model compile peak.
        with (out/'compile.log').open('w') as log:
            started=time.monotonic()
            process=subprocess.Popen(['/usr/bin/time','-v']+command,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=base)
            print(f'compile_started devices={count} pid={process.pid}',flush=True)
            while True:
                code=process.poll()
                status=dict(devices=count,pid=process.pid,elapsed_seconds=time.monotonic()-started,returncode=code)
                (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
                if code is not None:
                    print(json.dumps(status),flush=True)
                    failed=failed or code != 0
                    break
                time.sleep(5)
    return 1 if failed else 0


if __name__=='__main__':raise SystemExit(main())
