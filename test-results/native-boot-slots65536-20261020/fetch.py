"""Fetch experiment evidence; leave objects and executable on the server."""
from pathlib import Path
import io
import subprocess
import tarfile

root = Path(__file__).resolve().parent
remote = '''from pathlib import Path
import sys,tarfile
root=Path('/home/xuming/poseidon-native-boot-slots65536-20261020')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for p in sorted(root.rglob('*')):
  if p.is_file() and p.suffix in ('.json','.log','.cpp','.py'):
   archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
'''
result = subprocess.run(['ssh', '188Server', 'python3', '-'], input=remote.encode(),
                        stdout=subprocess.PIPE, check=True)
with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
    archive.extractall(root, filter='data')
print('Fetched evidence bytes:', len(result.stdout))
