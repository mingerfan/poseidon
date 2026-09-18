"""Run a fixed two-key SEAL diagnostic, never retry until success or change gates."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from hecate_python_env import ROOT,WORK,VENV,enter_nix,digest

BINARY=WORK/'build-dacapo/seal-golden-keys/seal_rotation_precision_probe'
SOURCE=ROOT/'scripts/baseline/seal_keys/rotation_precision_probe.cpp'

def main():
    if sys.argv[1:2]!=['--inside']:
        if Path.cwd().resolve()!=ROOT: raise ValueError('Requires source root')
        os.umask(0o077)
        root=Path(tempfile.mkdtemp(prefix='seal-rotation-precision-',dir=WORK/'results'))
        print('Rotation precision diagnostic: '+str(root),flush=True)
        command=shlex.join([str(VENV/'bin/python'),str(Path(__file__).resolve()),'--inside',str(root)])
        return enter_nix(command,seconds=180)
    root=Path(sys.argv[2]).resolve()
    if root.parent!=WORK/'results' or not root.name.startswith('seal-rotation-precision-'):
        raise ValueError('Invalid diagnostic output')
    sources={str(p):digest(p) for p in (SOURCE,Path(__file__),SOURCE.with_name('CMakeLists.txt'),BINARY)}
    with (root/'stages.jsonl').open('x') as output,(root/'stderr.log').open('x') as error:
        code=subprocess.run([str(BINARY)],stdout=output,stderr=error,timeout=150).returncode
    if code: raise RuntimeError('Native diagnostic failed; retained stderr')
    rows=[json.loads(line) for line in (root/'stages.jsonl').read_text().splitlines()]
    stages=['encode_decode','encrypted_input','modswitch_to_level1',
            'rotate_level1_1','rotate_level13_1','rotate_level1_2','rotate_level13_2',
            'rotate_level1_3','rotate_level13_3','sum_level1','sum_level13','sum_high_then_modswitch']
    if len(rows)!=72 or [r['stage'] for r in rows]!=stages*6:
        raise ValueError('Incomplete fixed diagnostic')
    summary={s:max(r['max_abs_first4'] for r in rows if r['stage']==s) for s in stages}
    report=dict(status='diagnostic_completed',purpose='localize_rotation_precision_not_acceptance',
                seal_version='4.0.0',polynomial_degree=32768,modulus_bits=[60]*14,security='tc128',
                scale_log2=40,trials=2,input_vectors=3,rows=rows,max_abs_first4_by_stage=summary,
                source_hashes=sources,stages_sha256=digest(root/'stages.jsonl'),
                original_failed_key_reproduced=False,keys_written_to_disk=False,
                compiler_parameters_changed=False,threshold_changed=False,agent_calls=0,
                registered_execution_backend=False)
    if any(digest(Path(p))!=h for p,h in sources.items()): raise ValueError('Diagnostic producer changed')
    (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
