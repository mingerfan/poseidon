"""Developer-only affine circuit probe. Parses a restricted AST; never executes DSL.
Only public graph/constants are used. It does not validate CKKS, compiler or runtime.
"""
import ast,math,itertools
from fractions import Fraction
from dataclasses import dataclass
def need(ok,code):
 if not ok:raise ValueError(code)
def number(v):
 need(type(v)in(int,float)and math.isfinite(v),"finite_public_number")
 return Fraction(v)if type(v)is int else Fraction.from_float(v)
def dimensions(value):
 if type(value)is list:
  need(bool(value),"empty_public_tensor");shapes=[dimensions(v)for v in value]
  need(all(s==shapes[0]for s in shapes),"ragged_public_tensor")
  return(len(value),)+shapes[0]
 return()
def flatten(value):
 return [v for item in value for v in flatten(item)]if type(value)is list else[number(value)]
def coords(shape):
 return itertools.product(*(range(n)for n in shape))
def flat_index(index,shape):
 out=0
 for i,n in zip(index,shape):out=out*n+i
 return out
@dataclass(frozen=True)
class Tensor:
 shape:tuple
 rows:tuple
@dataclass(frozen=True)
class Cipher:
 rows:tuple
@dataclass(frozen=True)
class Plain:
 values:tuple
def zero(width):return (Fraction(0),)*width
def scale(row,v):return tuple(x*v for x in row)
def combine(a,b,sign=1):return tuple(x+sign*y for x,y in zip(a,b))
def reference(request):
 model=request["model"];total=sum(math.prod(v["shape"])for v in model["inputs"]);width=total+1
 need(1<=total<=64 and len(model["nodes"])<=64,"probe_capacity")
 values={};offset=0
 for spec in model["inputs"]:
  shape=tuple(spec["shape"]);rows=[]
  for index in range(math.prod(shape)):
   row=list(zero(width));row[offset+index]=1;rows.append(tuple(row))
  values[spec["name"]]=Tensor(shape,tuple(rows));offset+=len(rows)
 for name,value in model["constants"].items():
  shape=dimensions(value);rows=[]
  for v in flatten(value):rows.append(zero(total)+(v,))
  values[name]=Tensor(shape,tuple(rows))
 for node in model["nodes"]:
  need(len(node["outputs"])==1,"probe_single_node_output")
  args=[values[n]for n in node["inputs"]];x=args[0];attrs=node["attrs"];op=node["op"]
  if op in("reshape","flatten"):
   shape=tuple(attrs["shape"])if op=="reshape"else(math.prod(x.shape),)
   need(all(type(n)is int and n>0 for n in shape)and math.prod(shape)==len(x.rows),"reshape_size")
   y=Tensor(shape,x.rows)
  elif op in("permute","transpose"):
   dims=tuple(attrs["dims"])if op=="permute"else None
   if dims is None:
    dims=list(range(len(x.shape)));a=attrs["dim0"]%len(dims);b=attrs["dim1"]%len(dims);dims[a],dims[b]=dims[b],dims[a];dims=tuple(dims)
   need(sorted(dims)==list(range(len(x.shape))),"permutation_axes")
   shape=tuple(x.shape[d]for d in dims);rows=[]
   for out in coords(shape):
    index=[0]*len(shape)
    for axis,d in enumerate(dims):index[d]=out[axis]
    rows.append(x.rows[flat_index(index,x.shape)])
   y=Tensor(shape,tuple(rows))
  elif op=="linear":
   need(len(args)in(2,3),"linear_arity");w=args[1]
   need(len(w.shape)==2 and w.shape[1]==x.shape[-1]and
        all(all(c==0 for c in row[:-1])for row in w.rows),"linear_public_weight_shape")
   kout,kin=w.shape;rows=[]
   bias=args[2]if len(args)==3 else None
   if bias is not None:need(bias.shape==(kout,)and all(all(c==0 for c in row[:-1])for row in bias.rows),"linear_bias_shape")
   for block in range(math.prod(x.shape[:-1])):
    for out in range(kout):
     row=zero(width)
     for k in range(kin):row=combine(row,scale(x.rows[block*kin+k],w.rows[out*kin+k][-1]))
     if bias is not None:row=combine(row,bias.rows[out])
     rows.append(row)
   y=Tensor(x.shape[:-1]+(kout,),tuple(rows))
  else:raise ValueError("unsupported_probe_graph_operator")
  need(len(y.rows)<=128,"probe_intermediate_capacity");values[node["outputs"][0]]=y
 rows=tuple(row for out in model["outputs"]for row in values[out["value"]].rows)
 need(1<=len(rows)<=64,"probe_output_capacity")
 return total,rows
def program(request,source,total):
 layout=request["layout"];period=layout["input_slot_period"];width=total+1
 need(period in(4,8,16,32,64),"probe_period")
 tree=ast.parse(source);need(len(source.encode())<=65536 and sum(1 for _ in ast.walk(tree))<=4096,"probe_source_capacity")
 need(len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef)and tree.body[0].name=="golden","probe_single_function")
 fun=tree.body[0];names=[v["dsl_name"]for v in layout["inputs"]]+["zero_ct"]
 need([a.arg for a in fun.args.args]==names and not fun.args.vararg and not fun.args.kwarg and not fun.args.defaults and not fun.args.kwonlyargs,"probe_entry_signature")
 bases={};base=0
 for spec in request["model"]["inputs"]:bases[spec["name"]]=base;base+=math.prod(spec["shape"])
 env={}
 for spec in layout["inputs"]:
  rows=[]
  for slot in range(period):
   row=list(zero(width))
   if slot<spec["elements"]:row[bases[spec["name"]]+spec["offset"]+slot]=1
   rows.append(tuple(row))
  env[spec["dsl_name"]]=Cipher(tuple(rows))
 env["zero_ct"]=Cipher(tuple(zero(width)for _ in range(period)))
 for name,value in request["public_constants"].items():
  if type(value)is list:need(len(value)==period,"public_constant_period");data=tuple(number(x)for x in value)
  else:data=(number(value),)*period
  env[name]=Plain(data)
 def expression(n):
  if isinstance(n,ast.Name):need(n.id in env,"unbound_probe_name");return env[n.id]
  if isinstance(n,ast.Constant):return Plain((number(n.value),)*period)
  if isinstance(n,ast.UnaryOp)and isinstance(n.op,ast.USub):
   x=expression(n.operand)
   return Cipher(tuple(scale(row,-1)for row in x.rows))if isinstance(x,Cipher)else Plain(tuple(-v for v in x.values))
  if isinstance(n,ast.BinOp)and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult)):
   a,b=expression(n.left),expression(n.right);need(isinstance(a,Cipher)or isinstance(b,Cipher),"probe_cipher_operand")
   if isinstance(n.op,ast.Mult):
    need(not(isinstance(a,Cipher)and isinstance(b,Cipher)),"non_affine_encrypted_multiply")
    c,p=(a,b)if isinstance(a,Cipher)else(b,a)
    return Cipher(tuple(scale(row,v)for row,v in zip(c.rows,p.values)))
   def rows(x):return x.rows if isinstance(x,Cipher)else tuple(zero(total)+(v,)for v in x.values)
   return Cipher(tuple(combine(x,y,-1 if isinstance(n.op,ast.Sub)else 1)for x,y in zip(rows(a),rows(b))))
  if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=="rotate":
   need(len(n.args)==1 and not n.keywords and isinstance(n.args[0],ast.Constant),"static_probe_rotation")
   k=n.args[0].value;need(type(k)is int and 0<k<period and k&(k-1)==0,"permitted_probe_rotation")
   x=expression(n.func.value);need(isinstance(x,Cipher),"cipher_probe_rotation")
   return Cipher(tuple(x.rows[(slot+k)%period]for slot in range(period)))
  if isinstance(n,(ast.List,ast.Tuple)):return tuple(expression(x)for x in n.elts)
  raise ValueError("unsupported_probe_source_syntax")
 need(fun.body and isinstance(fun.body[-1],ast.Return),"probe_final_return")
 for stmt in fun.body[:-1]:
  need(isinstance(stmt,ast.Assign)and len(stmt.targets)==1 and isinstance(stmt.targets[0],ast.Name),"probe_assignment")
  env[stmt.targets[0].id]=expression(stmt.value)
 returned=expression(fun.body[-1].value)
 if isinstance(returned,Cipher):returned=(returned,)
 need(type(returned)is tuple and len(returned)==layout["output_ciphertexts"]and all(isinstance(x,Cipher)for x in returned),"probe_output_binding")
 return tuple(returned[ct].rows[slot]for ct,slot in layout["output_selectors"])
def analyze(request,source):
 total,expected=reference(request);actual=program(request,source,total)
 need(len(actual)==len(expected),"probe_output_count")
 differences=[(abs(a-b),out,index)for out,(ar,er)in enumerate(zip(actual,expected))for index,(a,b)in enumerate(zip(ar,er))if a!=b]
 worst=max(differences,default=(Fraction(0),None,None))
 return dict(public_input_elements=total,public_output_elements=len(expected),
  exact_public_affine_match=not differences,coefficient_mismatches=len(differences),
  maximum_absolute_coefficient_difference=float(worst[0]),worst_output_index=worst[1],worst_input_index=worst[2],
  private_test_inputs_used=False,private_reference_outputs_used=False,compiled=False,encrypted_execution=False,
  scope="Exact rational arithmetic on serialized public coefficients; restricted affine graph/AST subset, not CKKS or full DSL acceptance.")
