#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import tarfile

base = Path('/home/xuming/poseidon-model-regression-20261021')
paths = [base/'traced/ResNet20.mlirbc', base/'qwen-prefix/trace-qwen-prefix.mlirbc',
         Path('/home/xuming/poseidon-waterline-20261013/qwen3-traced-bsgs/trace_qwen3.mlirbc')]
rows = []
for path in paths:
    s = path.stat()
    row = dict(path=str(path), bytes=s.st_size, mtime_ns=s.st_mtime_ns)
    if s.st_size < 1024**3: row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append(row)
binaries = [base/'hecate-opt-baseline',
            Path('/home/xuming/poseidon-threshold-rescale-20261021/toolchain/bin/hecate-opt'),
            Path('/home/xuming/poseidon-threshold-rescale-20261021/build/bin/poseidon_runtime_gpu_qwen_e2e')]
records = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries}
with tarfile.open(base/'qwen-source.tar.gz') as archive:
    original = archive.extractfile('test-results/model-regression-20261021/trace-qwen-prefix.py').read()
(base/'trace-qwen-prefix-kv-original.py').write_bytes(original)
signature = Path('/home/xuming/poseidon-waterline-20261013/qwen3-traced-bsgs/signature.json')
(base/'source-bindings.json').write_text(json.dumps(dict(inputs=rows, binaries=records,
    full_qwen_signature_sha256=hashlib.sha256(signature.read_bytes()).hexdigest()), indent=2)+'\n')
