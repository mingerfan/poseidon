"""Retained-file byte accounting while workers legitimately delete transient keys.

Only disappearance is tolerated. Permission and other I/O failures still abort.
No file contents are read; integrity of persistent evidence is checked separately.
"""
import os,stat
from pathlib import Path

def size_bytes(paths):
 total=0
 for root in set(map(Path,paths)):
  pending=[root]
  while pending:
   directory=pending.pop()
   try:
    with os.scandir(directory) as entries:
     for entry in entries:
      try:
       info=entry.stat(follow_symlinks=False)
       if stat.S_ISDIR(info.st_mode):pending.append(Path(entry.path))
       elif stat.S_ISREG(info.st_mode):total+=info.st_size
       elif stat.S_ISLNK(info.st_mode):
        # Match earlier accounting of file links; never recurse through a
        # directory link. A missing temporary link contributes no bytes.
        target=entry.stat(follow_symlinks=True)
        if stat.S_ISREG(target.st_mode):total+=target.st_size
      except FileNotFoundError:
       continue
   except FileNotFoundError:
    continue
 return total
