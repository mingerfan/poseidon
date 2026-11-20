#!/usr/bin/env python3
"""External compiler-owned ciphertext-state evaluation entry."""
from pathlib import Path
import json,os,sys
os.environ.setdefault('POSEIDON_PLATFORM','aarch64-linux')
# SDK selection precedes every backend import, including inside the pure shell.
if '--profile' in sys.argv:
    index=sys.argv.index('--profile')+1
    value=json.loads(Path(sys.argv[index]).read_bytes())
    os.environ['POSEIDON_SDK_MANIFEST']=value['selector']['path']
elif '--selector' in sys.argv:
    os.environ['POSEIDON_SDK_MANIFEST']=str(Path(sys.argv[sys.argv.index('--selector')+1]).absolute())
sys.path.insert(0,str(Path(__file__).resolve().parent/'baseline'))
from compiler_managed_cli import main
if __name__=='__main__':raise SystemExit(main())
