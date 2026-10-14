"""Inactive transport draft; never imported by the application or package runner."""
import json, math, hashlib, sys
from pathlib import Path
CODEC_ROOT=Path(__file__).resolve().parent.parent/"generic_provenance_dag_v1"
sys.path.insert(0,str(CODEC_ROOT))
from codec import encode,decode,measure,raw,require,FORMAT as DAG_FORMAT
FORMAT="poseidon-provenance-transport-draft-v1"
PURPOSES=frozenset(("task-metadata","model-provenance","capture-reference","decomposition"))
WIRE_LIMIT=4*1024**2
LOGICAL_LIMIT=16*1024**2
def canonical(value):
 return raw(value,sorted_keys=True)
def logical_digest(value):
 return hashlib.sha256(canonical(value)).hexdigest()
def purpose(kind):
 require(type(kind)is str and kind in PURPOSES,"transport_purpose")
def serialize(value,kind,*,limit=WIRE_LIMIT):
 purpose(kind)
 require(type(value)is dict and value.get("format")not in (DAG_FORMAT,FORMAT),"transport_logical_root")
 require(type(limit)is int and 1<=limit<=WIRE_LIMIT,"transport_wire_policy")
 measure(value,LOGICAL_LIMIT)
 plain=raw(value)
 if len(plain)<=limit:
  return plain
 wrapper=dict(format=FORMAT,purpose=kind,value=value)
 packet=encode(wrapper)
 encoded=raw(packet)
 require(len(encoded)<=limit,"transport_wire_capacity")
 return encoded
def parse(data,kind,*,limit=WIRE_LIMIT):
 purpose(kind)
 require(type(data)is bytes and type(limit)is int and 1<=limit<=WIRE_LIMIT
         and 0<len(data)<=limit,"transport_wire_budget")
 def pairs(items):
  result={}
  for key,value in items:
   require(key not in result,"transport_duplicate_key")
   result[key]=value
  return result
 def nonfinite(_):raise ValueError("transport_nonfinite")
 def real(token):
  v=float(token);require(math.isfinite(v),"transport_nonfinite");return v
 try:value=json.loads(data,object_pairs_hook=pairs,parse_constant=nonfinite,parse_float=real)
 except (RecursionError,UnicodeDecodeError,MemoryError) as e:
  raise ValueError("transport_parse_resource")from e
 require(type(value)is dict,"transport_wire_root")
 if value.get("format")==DAG_FORMAT:
  wrapper=decode(value)
  require(type(wrapper)is dict and set(wrapper)=={"format","purpose","value"}
          and wrapper["format"]==FORMAT and wrapper["purpose"]==kind,"transport_wrapper_scope")
  value=wrapper["value"]
 require(type(value)is dict and value.get("format")not in (DAG_FORMAT,FORMAT),"transport_logical_root")
 measure(value,LOGICAL_LIMIT)
 return value
