import ast,json
from pathlib import Path
from test_capacity_shallow_packing import current
from unified_graph_lowering import candidate_source
for name in ('rank_axis_8','rank_linear_16'):
 _,q=current(name)
 try:source,info=candidate_source(q)
 except ValueError as e:
  tb=e.__traceback__;source=None
  while tb:
   if tb.tb_frame.f_code.co_name=='lower' and 'source' in tb.tb_frame.f_locals:source=tb.tb_frame.f_locals['source']
   tb=tb.tb_next
  print(name,'FAILED',str(e),flush=True)
 else:print(name,'ACCEPTED',info,flush=True)
 if source:
  dest=Path(__file__).parent/(name+'-initial-source.py');dest.write_text(source)
  tree=ast.parse(source)
  print(name,'ast',len(list(ast.walk(tree))),'bytes',len(source),'lines',len(source.splitlines()))
  print('largest',sorted([(len(list(ast.walk(st))),st.lineno,ast.unparse(st)[:190])for f in tree.body for st in f.body],reverse=True)[:6])
 print('packing', {k:v for k,v in q['constant_origins']['capacity_tensor_packing'].items() if k not in ('bindings','linear_tiles')})
