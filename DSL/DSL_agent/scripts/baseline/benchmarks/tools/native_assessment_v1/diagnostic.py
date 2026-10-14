import sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path("scripts/baseline").resolve()))
from task_numerical_assessment import implementation
try:print(implementation())
except Exception:traceback.print_exc()
