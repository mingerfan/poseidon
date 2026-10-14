"""Inactive draft: lossless ordered JSON provenance without duplicate containers."""
import hashlib,json,math
FORMAT="poseidon-provenance-container-dag-draft-v1"
WIRE_BYTES=4*1024**2
LOGICAL_BYTES=16*1024**2
MAX_NODES=65536
MAX_DEPTH=64
def raw(v,*,sorted_keys=False):
 return json.dumps(v,separators=(",",":"),sort_keys=sorted_keys,allow_nan=False).encode()
def sha(v,*,sorted_keys=False):return hashlib.sha256(raw(v,sorted_keys=sorted_keys)).hexdigest()
def require(condition,code):
 if not condition:raise ValueError(code)
def atom(v):
 require(type(v)in(str,int,float,bool,type(None)),"json_atom_type")
 require(type(v)is not float or math.isfinite(v),"nonfinite_atom")
 if type(v)is str:require(len(v)<=WIRE_BYTES and len(raw(v))<=WIRE_BYTES,"atom_budget")
 if type(v)is int:require(v.bit_length()<=4096,"integer_budget")
 return v
def measure(value,limit,*,depth_limit=MAX_DEPTH):
 """Count ordered JSON bytes before serialization or reconstruction."""
 memo={};visiting=set()
 def count(v,depth):
  if type(v)not in(dict,list):
   atom(v);return len(raw(v)),0
  require(depth<depth_limit,"provenance_depth")
  marker=id(v);require(marker not in visiting,"cyclic_provenance")
  if marker in memo:
   length,nested=memo[marker];require(depth+nested<=depth_limit,"provenance_depth");return length,nested
  visiting.add(marker);length=2+max(0,len(v)-1);nested=1
  try:
   if type(v)is dict:
    require(all(type(k)is str for k in v),"json_key_type")
    entries=v.items()
   else:entries=((None,x)for x in v)
   for k,x in entries:
    if k is not None:atom(k);length+=len(raw(k))+1
    size,child_depth=count(x,depth+1);length+=size;nested=max(nested,child_depth+1)
    require(length<=limit,"json_size_budget")
  finally:visiting.remove(marker)
  require(length<=limit,"json_size_budget");memo[marker]=(length,nested);return length,nested
 return count(value,0)[0]
def encode(value,*,logical_limit=LOGICAL_BYTES):
 require(type(logical_limit)is int and 1<=logical_limit<=LOGICAL_BYTES,"logical_budget")
 require(type(value)is dict,"provenance_root")
 logical_size=measure(value,logical_limit)
 nodes=[];known={};visiting=set()
 def visit(v,depth):
  if type(v)not in(dict,list):return atom(v)
  require(depth<MAX_DEPTH,"provenance_depth")
  marker=id(v);require(marker not in visiting,"cyclic_provenance");visiting.add(marker)
  try:
   if type(v)is list:item=["list",[visit(x,depth+1)for x in v]]
   else:
    require(all(type(k)is str for k in v),"json_key_type")
    item=["dict",[[k,visit(x,depth+1)]for k,x in v.items()]]
  finally:visiting.remove(marker)
  key=raw(item)
  if key not in known:
   require(len(nodes)<MAX_NODES,"provenance_node_budget");known[key]=len(nodes);nodes.append(item)
  return ["ref",known[key]]
 root=visit(value,0)
 packet=dict(format=FORMAT,root=root,nodes=nodes,logical_bytes=logical_size,
             canonical_sha256=sha(value,sorted_keys=True),ordered_sha256=sha(value),
             production_active=False)
 packet["binding"]=sha(packet,sorted_keys=True)
 measure(packet,WIRE_BYTES,depth_limit=MAX_DEPTH+8)
 return packet
def decode(packet,*,logical_limit=LOGICAL_BYTES):
 require(type(logical_limit)is int and 1<=logical_limit<=LOGICAL_BYTES,"logical_budget")
 fields={"format","root","nodes","logical_bytes","canonical_sha256","ordered_sha256","production_active","binding"}
 require(type(packet)is dict and set(packet)==fields and packet["format"]==FORMAT
         and packet["production_active"]is False,"provenance_packet")
 measure(packet,WIRE_BYTES,depth_limit=MAX_DEPTH+8)
 require(packet["binding"]==sha({k:v for k,v in packet.items()if k!="binding"},sorted_keys=True),"packet_binding")
 nodes=packet["nodes"];require(type(nodes)is list and 1<=len(nodes)<=MAX_NODES,"provenance_node_budget")
 sizes=[];depths=[];known=set()
 def token(t,position):
  if type(t)is list:
   require(len(t)==2 and t[0]=="ref" and type(t[1])is int and 0<=t[1]<position,"provenance_reference")
   return sizes[t[1]],depths[t[1]]
  atom(t);return len(raw(t)),0
 for i,item in enumerate(nodes):
  require(type(item)is list and len(item)==2 and item[0]in("list","dict")and type(item[1])is list,"provenance_node")
  encoded=raw(item);require(encoded not in known,"noncanonical_duplicate_node");known.add(encoded)
  children=item[1];size=2+max(0,len(children)-1);depth=1
  if item[0]=="dict":
   names=set()
   for pair in children:
    require(type(pair)is list and len(pair)==2 and type(pair[0])is str and pair[0]not in names,"provenance_key")
    names.add(pair[0]);size+=len(raw(pair[0]))+1
    length,nested=token(pair[1],i);size+=length;depth=max(depth,nested+1)
    require(size<=logical_limit,"expanded_provenance_budget")
  else:
   for t in children:
    length,nested=token(t,i);size+=length;depth=max(depth,nested+1)
    require(size<=logical_limit,"expanded_provenance_budget")
  require(depth<=MAX_DEPTH and size<=logical_limit,"provenance_depth_or_size")
  sizes.append(size);depths.append(depth)
 root=packet["root"]
 require(type(root)is list and len(root)==2 and root[0]=="ref" and type(root[1])is int
         and 0<=root[1]<len(nodes)and nodes[root[1]][0]=="dict","provenance_root")
 require(type(packet["logical_bytes"])is int and packet["logical_bytes"]==sizes[root[1]]<=logical_limit,"logical_size_claim")
 reachable=set();pending=[root[1]]
 while pending:
  i=pending.pop()
  if i in reachable:continue
  reachable.add(i);kind,entries=nodes[i];tokens=entries if kind=="list"else[p[1]for p in entries]
  pending.extend(t[1]for t in tokens if type(t)is list)
 require(len(reachable)==len(nodes),"unreachable_provenance_nodes")
 def expand_token(t):
  if type(t)is not list:return t
  kind,entries=nodes[t[1]]
  # Fresh containers per occurrence prevent aliases between independently stored records.
  if kind=="list":return[expand_token(x)for x in entries]
  return{k:expand_token(x)for k,x in entries}
 value=expand_token(root)
 require(sha(value)==packet["ordered_sha256"] and sha(value,sorted_keys=True)==packet["canonical_sha256"],"logical_content_binding")
 require(len(raw(value))==packet["logical_bytes"],"logical_size_claim")
 return value
