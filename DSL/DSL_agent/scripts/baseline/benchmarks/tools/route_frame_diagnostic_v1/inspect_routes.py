"""Developer-only symbolic route audit. Parses public candidate AST without executing it."""
import ast,hashlib,json,math,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(x,code):
 if not x:raise ValueError(code)
def bound(d):require(d["binding"]==digest({k:v for k,v in d.items()if k!="binding"}),"binding")
def add(a,b):
 c=dict(a)
 for k,v in b.items():
  c[k]=c.get(k,0.)+v
  if c[k]==0.:del c[k]
 return c
def mul(a,b):
 c={}
 for k,v in a.items():
  for l,w in b.items():
   key=tuple(sorted(k+l));require(len(key)<=2,"degree_bound")
   c[key]=c.get(key,0.)+v*w
 return {k:v for k,v in c.items()if v!=0.}
class Frame:
 def __init__(self,p,env):self.p=p;self.env=env
 def evaluate(self,n):
  if isinstance(n,ast.Name):
   require(n.id in self.env,"unknown_symbol");return self.env[n.id]
  if isinstance(n,ast.Constant)and type(n.value)in(int,float):
   require(math.isfinite(n.value),"finite_literal");return [{():float(n.value)}if n.value else {}for _ in range(self.p)]
  if isinstance(n,ast.UnaryOp)and isinstance(n.op,(ast.UAdd,ast.USub)):
   a=self.evaluate(n.operand);return a if isinstance(n.op,ast.UAdd)else [{k:-v for k,v in x.items()}for x in a]
  if isinstance(n,ast.BinOp)and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult)):
   a,b=self.evaluate(n.left),self.evaluate(n.right)
   if isinstance(n.op,ast.Mult):return [mul(x,y)for x,y in zip(a,b)]
   if isinstance(n.op,ast.Sub):b=[{k:-v for k,v in x.items()}for x in b]
   return [add(x,y)for x,y in zip(a,b)]
  if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=="rotate"and len(n.args)==1 and not n.keywords:
   x=n.args[0]
   if isinstance(x,ast.Constant)and type(x.value)is int:step=x.value
   elif isinstance(x,ast.UnaryOp)and isinstance(x.op,ast.USub)and isinstance(x.operand,ast.Constant)and type(x.operand.value)is int:step=-x.operand.value
   else:raise ValueError("nonliteral_rotation")
   a=self.evaluate(n.func.value);return [a[(j+step)%self.p]for j in range(self.p)]
  raise ValueError("unsupported_ast:"+type(n).__name__)
 def statements(self,body,stop):
  for n in body:
   require(isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name),"straight_line_assignment")
   name=n.targets[0].id;self.env[name]=self.evaluate(n.value)
   if name==stop:return
  raise ValueError("missing_selected_assignment")
def basis(p,n,offset=0):return [{(offset+j,):1.}if j<n else {}for j in range(p)]
def coefficients(vector,p):
 if type(vector)in(int,float):vector=[vector]*p
 require(isinstance(vector,list)and len(vector)==p and all(type(x)in(int,float)and math.isfinite(x)for x in vector),"public_vector")
 return [{():float(x)}if x else {}for x in vector]
def coordinate(index,shape):
 c=[]
 for d in reversed(shape):c.append(index%d);index//=d
 return tuple(reversed(c))
def flat(c,shape):
 out=0
 for x,d in zip(c,shape):out=out*d+x
 return out
def permutation_indices(shape,dims):
 target=[shape[i]for i in dims];out=[]
 for j in range(math.prod(shape)):
  tc=coordinate(j,target);c=[0]*len(shape)
  for axis,i in enumerate(dims):c[i]=tc[axis]
  out.append(flat(c,shape))
 return out
def compare(a,b,n):
 require(len(a)==len(b),"carrier_length")
 mismatch=0;maximum=0.
 for x,y in zip(a[:n],b[:n]):
  err=max((abs(x.get(k,0.)-y.get(k,0.))for k in set(x)|set(y)),default=0.)
  mismatch+=err>1e-12;maximum=max(maximum,err)
 return dict(compared_slots=n,mismatched_slots=mismatch,maximum_coefficient_error=maximum,passed=mismatch==0)
def inspect(scope):
 bound(scope)
 for name,want in scope["evidence_files"].items():require(sha(name)==want,"evidence_hash")
 request=json.loads(Path(scope["request"]).read_bytes());response=json.loads(Path(scope["response"]).read_bytes())
 candidate=json.loads(response["raw"]);src=candidate["hecate_source"]
 require(candidate["request_id"]==request["request_id"],"candidate_request_binding")
 tree=ast.parse(src);require(sum(1 for _ in ast.walk(tree))<=4096,"ast_bound")
 fn=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==scope["function"]]
 require(len(fn)==1,"selected_function")
 layout=request["layout"];p=layout["input_slot_period"]
 inputs=request["model"]["inputs"];require(len(inputs)==2,"selected_two_input_broadcast")
 shape=inputs[0]["shape"];small=inputs[1]["shape"];n,m=math.prod(shape),math.prod(small)
 require(small==shape[-len(small):]and n<=p and m<=p,"suffix_broadcast_shape")
 env={k:coefficients(v,p)for k,v in request["public_constants"].items()}
 env.update(x0=basis(p,n),x1=basis(p,m,n),zero_ct=[{}for _ in range(p)])
 evaluator=Frame(p,env);evaluator.statements(fn[0].body,scope["permute_assignment"])
 expected_g=[{(j,):1.,(n+j%m,):1.}if j<n else {}for j in range(p)]
 node=next(x for x in request["model"]["nodes"]if x["id"]==scope["permute_node"])
 require(node["op"]=="permute","registered_permutation")
 indices=permutation_indices(shape,node["attrs"]["dims"])
 expected_pf=[expected_g[indices[j]]if j<n else {}for j in range(p)]
 g=compare(env[scope["broadcast_assignment"]],expected_g,n)
 pf=compare(env[scope["permute_assignment"]],expected_pf,n)
 origins=request["constant_origins"]["capacity_tensor_packing"]["bindings"]
 routes=[x for x in origins if x["node_id"]==node["id"]]
 matched=[]
 expr=next(x.value for x in fn[0].body if isinstance(x,ast.Assign)and isinstance(x.targets[0],ast.Name)and x.targets[0].id==scope["permute_assignment"])
 def terms(n):return terms(n.left)+terms(n.right)if isinstance(n,ast.BinOp)and isinstance(n.op,ast.Add)else[n]
 for term in terms(expr):
  require(isinstance(term,ast.BinOp)and isinstance(term.op,ast.Mult)and isinstance(term.left,ast.Name),"selected_mask_times_rotation")
  name=term.left.id;arg=term.right;net=0
  while isinstance(arg,ast.Call)and isinstance(arg.func,ast.Attribute)and arg.func.attr=="rotate":
   require(len(arg.args)==1 and isinstance(arg.args[0],ast.Constant)and type(arg.args[0].value)is int,"selected_constant_rotation")
   net+=arg.args[0].value;arg=arg.func.value
  require(isinstance(arg,ast.Name)and arg.id==scope["broadcast_assignment"],"selected_source_carrier")
  choices=[x for x in routes if x["key"]==name and x["step"]==net%p]
  require(len(choices)==1,"unique_registered_step")
  route=choices[0]
  matched.append(dict(binding=name,step=net%p,baby_step=route.get("baby_step"),giant_step=route.get("giant_step"),
     registered_step_matches=True,mask_applied_without_outer_giant_rotation=True))
 out=dict(format="poseidon-public-route-frame-diagnostic-v1",status="diagnosed"if not pf["passed"]else"passed",
  parent_binding=scope["binding"],request_id=request["request_id"],candidate_source_sha256=hashlib.sha256(src.encode()).hexdigest(),
  broadcast=g,permutation=pf,registered_route_terms=matched,
  conclusion="Broadcast carrier matches the declared suffix mapping; permutation has a coordinate-frame mismatch despite matching net route steps."if g["passed"]and not pf["passed"]else"See carrier comparisons.",
  evidence_level="static_public_symbolic_coefficients",private_test_inputs_read=False,reference_arrays_read=False,
  candidate_executed=False,new_api_calls=0,new_compiler_calls=0,new_encrypted_calls=0,
  original_candidate_changed=False,backend_changed=False)
 out["binding"]=digest(out)
 with (HERE/"report.json").open("xb")as f:os.chmod(HERE/"report.json",0o600);f.write(canonical(out))
 print(json.dumps({k:out[k]for k in ["status","binding","broadcast","permutation","new_api_calls","new_compiler_calls","new_encrypted_calls"]}))
if __name__=="__main__":
 require(sys.argv[1:]==["--inspect"],"mode");inspect(json.loads((HERE/"scope.json").read_bytes()))
