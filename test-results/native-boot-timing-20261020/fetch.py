#!/usr/bin/env python3
"""Fetch compact benchmark evidence, excluding binaries and build objects."""
import io
from pathlib import Path
import subprocess
import tarfile

root=Path(__file__).resolve().parent
remote=r'''
from pathlib import Path
import sys,tarfile
root=Path('/home/xuming/poseidon-native-boot-timing-20261020')
files=[]
for p in root.rglob('*'):
    if p.is_file() and p.suffix in ('.json','.log','.py','.sh','.cpp'):
        if p.name!='standalone-compat.cpp':files.append(p)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
    for p in files:archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
'''
r=subprocess.run(['ssh','188Server','python3','-'],input=remote.encode(),stdout=subprocess.PIPE,check=True)
with tarfile.open(fileobj=io.BytesIO(r.stdout)) as archive:archive.extractall(root,filter='data')
print('Fetched compact bytes',len(r.stdout))
