"""Measure Joint ratio128/scale60 correction, retaining degree58."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'resnet18-precision-v1/component'


if __name__=='__main__':
    out = ROOT/'component'
    out.mkdir()
    source = out/'probe.cpp'
    text = (PREVIOUS/'probe.cpp').read_text()
    old = '        profile_config.bootstrap_ratio = static_cast<std::uint32_t>(std::stoul(std::getenv("PRECISION_PROBE_RATIO")));\n        profile_config.profile_id += "-ratio"+std::to_string(profile_config.bootstrap_ratio);'
    assert text.count(old)==1
    text = text.replace(old,'        profile_config.eval_mod_log_scale = 60;\n        profile_config.bootstrap_ratio = 128;\n'
        '        profile_config.profile_id = "poseidon-gpu-degree58-da3-full-cosine-scale60-ratio128-real";')
    text = text.replace('0, 5, 3, 25, 0, 59);','0, 7, 3, 25, 0, 59);')
    source.write_text(text)
    old = json.loads((PREVIOUS/'provenance.json').read_text())
    command,link = old['compile'][:],old['link'][:]
    original_object = command[command.index('-o')+1]
    command[command.index('-c')+1] = str(source)
    command[command.index('-o')+1] = str(out/'probe.o')
    link[link.index(original_object)] = str(out/'probe.o')
    binary = out/'probe'
    link[link.index('-o')+1] = str(binary)
    with (out/'build.log').open('x') as log:
        for args in (command,link):
            subprocess.run(args,cwd=old['cwd'],stdout=log,stderr=subprocess.STDOUT,check=True)
    sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'provenance.json').write_text(json.dumps(dict(compile=command,link=link,cwd=old['cwd'],
        source_sha256=sha(source),binary_sha256=sha(binary),change='Joint correction: ratio128 reduces systematic bias; scale60 reduces GPU arithmetic noise'),indent=2)+'\n')
    uuid = 'GPU-5aaf51d1-a1db-e8f5-19bd-db86599054c5'
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
        '--format=csv,noheader'],text=True)
    assert uuid not in processes,'GPU1 occupied'
    env = {k:v for k,v in os.environ.items() if not k.startswith(('POSEIDON_','CUDA_'))}
    env.update(CUDA_VISIBLE_DEVICES=uuid,POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='31744',
        POSEIDON_RUNTIME_BOOTSTRAP_DEGREE='65536',POSEIDON_RUNTIME_BOOTSTRAP_LINEAR_MODE='no_hoist',
        POSEIDON_RUNTIME_BOOTSTRAP_FULL_COSINE='1',POSEIDON_RUNTIME_BOOTSTRAP_MODEL_PROFILE='1',
        POSEIDON_RUNTIME_BOOTSTRAP_REAL_PROFILE='1',POSEIDON_RUNTIME_BOOTSTRAP_DIAGNOSTICS='1',
        PRECISION_PROBE_OUTPUT=str(out/'combined.json'))
    (ROOT/'component.status').write_text('COMBINED_COMPONENT_RUNNING\n')
    with (out/'combined.log').open('x') as log:
        subprocess.run([str(binary)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/'combined.log').open('a') as log:
        log.write('NATIVE_NOHOIST_PROBE_EXIT=0\n')
    (ROOT/'component.status').write_text('COMBINED_COMPONENT_PASS\n')
