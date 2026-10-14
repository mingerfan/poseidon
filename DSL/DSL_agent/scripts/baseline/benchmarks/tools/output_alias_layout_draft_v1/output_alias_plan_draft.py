"""Data-only draft; no application or backend integration."""
import copy,hashlib,json,math
FORMAT="poseidon-exact-output-alias-plan-draft-v1"
def require(ok,code):
 if not ok:raise ValueError(code)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def build(outputs,shapes,*,period,logical_element_limit,physical_ciphertext_limit,rank_limit,model_binding):
 require(type(outputs)is list and 1<=len(outputs)<=4,"output_count")
 require(type(shapes)is dict and type(period)is int and 0<period<=4096,"source_shapes_and_period")
 require(type(model_binding)is str and len(model_binding)==64 and all(c in"0123456789abcdef"for c in model_binding),"model_binding")
 require(all(type(v)is int and v>0 for v in(logical_element_limit,physical_ciphertext_limit,rank_limit)),"limits")
 names=set();declared=[];total=0
 for row in outputs:
  require(type(row)is dict and set(row)=={"name","value"}and
   all(type(row[k])is str and row[k] for k in row),"output_fields")
  name,ref=row["name"],row["value"]
  require(name not in names and ref in shapes,"bound_named_output");names.add(name)
  shape=shapes[ref]
  require(type(shape)is list and len(shape)<=rank_limit and all(type(d)is int and d>0 for d in shape),"shape")
  size=math.prod(shape);total+=size
  require(total<=logical_element_limit,"logical_elements")
  declared.append(dict(name=name,value=ref,shape=copy.deepcopy(shape),elements=size))
 first={};physical=[];logical=[];selectors=[]
 for item in declared:
  ref=item["value"]
  if ref not in first:
   chunks=[]
   for offset in range(0,item["elements"],period):
    require(len(physical)<physical_ciphertext_limit,"physical_ciphertexts")
    row=dict(ciphertext=len(physical),offset=offset,elements=min(period,item["elements"]-offset))
    chunks.append(row);physical.append(dict(source_value=ref,first_output_name=item["name"],**row))
   first[ref]=(item["name"],chunks)
  owner,chunks=first[ref]
  logical.append(dict(name=item["name"],source_value=ref,shape=item["shape"],
   alias_of=None if owner==item["name"] else owner,chunks=copy.deepcopy(chunks)))
  for chunk in chunks:selectors.extend([[chunk["ciphertext"],i]for i in range(chunk["elements"])])
 result=dict(format=FORMAT,model_binding=model_binding,source_outputs=copy.deepcopy(outputs),
  source_shapes={r["value"]:copy.deepcopy(shapes[r["value"]])for r in outputs},
  period=period,logical_element_limit=logical_element_limit,physical_ciphertext_limit=physical_ciphertext_limit,
  rank_limit=rank_limit,logical_outputs=logical,physical_outputs=physical,selectors=selectors,
  logical_elements=total,physical_ciphertexts=len(physical))
 result["binding"]=digest(result);return result
def check(record,outputs,shapes,**kwargs):
 expected=build(outputs,shapes,**kwargs)
 require(record==expected,"alias_plan_changed")
 return expected
