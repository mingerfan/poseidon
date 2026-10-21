#!/usr/bin/env python3
"""Copy compact profiling evidence; leave build products and SQLite on server."""
import io
from pathlib import Path
import subprocess
import tarfile

root=Path(__file__).resolve().parent
remote=r'''
from pathlib import Path
import sys,tarfile
root=Path('/home/xuming/poseidon-qwen-low-util-20261020')
files=[p for p in root.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.log','.csv','.cpp')]
for name in ('control','api-probe','api-probe-late','nsys','nsys-late','build-host-markers'):
    files.extend(p for p in (root/name).glob('*') if p.is_file()
        and p.suffix in ('.json','.jsonl','.log','.csv') and p.name!='decoded.json')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
    for path in files:tar.add(path,arcname=str(path.relative_to(root)),recursive=False)
'''
result=subprocess.run(['ssh','188Server','python3','-'],input=remote.encode(),stdout=subprocess.PIPE,check=True)
with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:tar.extractall(root,filter='data')
print('Fetched compact bytes',len(result.stdout))
