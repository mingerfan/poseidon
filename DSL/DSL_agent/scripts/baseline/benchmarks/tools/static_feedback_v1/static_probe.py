import sys,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path("scripts/baseline").resolve()))
from test_static_work_feedback import *
q=prepare_task(tiny_model())
print("LAYOUT",q["layout"]["inputs"],flush=True)
for cost in (1023,1025):
 o=validate_program(response(q,work_source(q,cost)),q)
 print("CHECK",cost,o,flush=True)
s='@hc.func("c,c")\\ndef golden(x0,zero_ct):\\n    return [x0*x0]\\n'
print("VALID",validate_program(response(q,s),q),flush=True)
