"""Run the selected local CTest commands against the same binary on 188Server."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve().parent
local = '/mnt/zhitai/KnowledgeLib/Research/FHE-GPU/poseidon/third_party/ckks-runtime/third_party/dacapo'
source = here/'runtime/third_party/dacapo'
compiler = here/'bin/hecate-opt'
expected = (here/'compiler-sha256.txt').read_text().strip()
assert hashlib.sha256((here/'hecate-opt.real').read_bytes()).hexdigest() == expected
tests = json.loads((here/'ctest-commands.json').read_text())['tests']
results = []
for test in tests:
    if test['name'] == 'DacapoFrontendBytecode':
        continue
    command = []
    for arg in test['command']:
        if 'python3' in arg and arg.startswith('/nix/store/'):
            arg = sys.executable
        elif arg == local+'/build/nix/bin/hecate-opt':
            arg = str(compiler)
        elif arg == local+'/build/nix/bin/DacapoModulusChainTest':
            arg = '/home/xuming/poseidon-local-scale-step3-20261031/bin/DacapoModulusChainTest'
        elif arg.startswith(local):
            arg = str(source)+arg[len(local):]
        command.append(arg)
    run = subprocess.run(command, capture_output=True, text=True)
    results.append(dict(name=test['name'], returncode=run.returncode,
                        command=command, stdout=run.stdout, stderr=run.stderr))
    print(test['name'], run.returncode, flush=True)
(here/'remote-regressions.json').write_text(json.dumps(results, indent=2)+'\n')
assert len(results) == 20 and all(x['returncode'] == 0 for x in results)
