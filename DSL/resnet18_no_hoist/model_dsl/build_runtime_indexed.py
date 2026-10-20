"""Build only extended test entrances, linking existing verified runtime libraries.

Does not modify the framework tree or replace existing binaries. The production
SequentialRuntime, compiler, cryptography and Native Boot libraries are reused.
"""
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRAME = ROOT.parent/'resnet20-framework-v1'
BUILD = FRAME/'build-runtime-v2/poseidon'


if __name__ == '__main__':
    out = ROOT/'runtime-build-indexed'
    out.mkdir()
    provenance = []
    targets = (
        ('poseidon_runtime_gpu_mlp_e2e', BUILD/'src/poseidon/runtime_api', 'poseidon_gpu_mlp_e2e_test.cpp'),
    )
    with (out/'build.log').open('x') as log:
        for name, cwd, source in targets:
            cmake = cwd/f'CMakeFiles/{name}.dir'
            flags = {}
            for line in (cmake/'flags.make').read_text().splitlines():
                if line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')):
                    key, value = line.split('=', 1)
                    flags[key.strip()] = shlex.split(value.strip())
            src = ROOT/'runtime-support'/source
            obj = out/(source+'.o')
            command = ['/usr/bin/c++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'],
                       *flags['CXX_FLAGS'], '-I'+str(FRAME/'poseidon/src/poseidon/tests/runtime_api'),
                       '-I'+str(ROOT/'runtime-support'), '-c', str(src), '-o', str(obj)]
            subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
            link = shlex.split((cmake/'link.txt').read_text())
            objects = [i for i, value in enumerate(link) if value.endswith('.cpp.o')]
            assert len(objects) == 1
            link[objects[0]] = str(obj)
            link[link.index('-o')+1] = str(out/name)
            subprocess.run(link, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
            provenance.append(dict(target=name,compile=command,link=link,cwd=str(cwd),
                                   source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                                   binary_sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('RUNTIME_TEST_ENTRANCES_BUILD_PASS', flush=True)
