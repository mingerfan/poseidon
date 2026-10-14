"""Data-only raw state and divisor admission for explicit source normalization."""
import re
from benchmark_graph import require
from pytorch_exact_arithmetic import classification,arguments,_roles
TYPES={"<f8","<i4","<i8","|b1"}
FLOAT_POLICIES={"strict-float64-v1","convert-float64-v1"}
FIELDS={"shape","dtype","sha256","normalized_float64_sha256"}
def validate_state(states,*,float_policy="strict-float64-v1"):
 require(type(float_policy)is str and float_policy in FLOAT_POLICIES,"Explicit public-state float policy")
 allowed=TYPES|({"<f4"}if float_policy=="convert-float64-v1"else set())
 require(type(states)is dict and all(type(n)is str and n for n in states),"Public state inventory")
 for value in states.values():
  require(type(value)is dict and set(value)==FIELDS and value["dtype"] in allowed,
   "Public state exact raw type")
  require(type(value["shape"])is list and all(type(d)is int and d>0 for d in value["shape"]),
   "Public state static shape")
  require(all(type(value[h])is str and re.fullmatch("[0-9a-f]{64}",value[h])
   for h in ("sha256","normalized_float64_sha256")),"Public state digest")
 return states
def check_original(states,declarations,*,float_policy="strict-float64-v1"):
 validate_state(states,float_policy=float_policy)
 require(type(declarations)is dict and set(states)==set(declarations),"Original public state names")
 for n,value in states.items():
  require(value["dtype"]==declarations[n]["dtype"] and value["shape"]==declarations[n]["shape"],
   "Original public state raw dtype/shape")
 return states
def float64_state(states,*,float_policy):
 """State.double() changes floating tensors only; raw integer/bool identity stays exact."""
 validate_state(states,float_policy=float_policy)
 result={}
 for name,value in states.items():
  result[name]=dict(value)
  if value["dtype"]=="<f4":
   result[name].update(dtype="<f8",sha256=value["normalized_float64_sha256"])
 return result
def check_normalization(states,normalized,*,float_policy):
 expected=float64_state(states,float_policy=float_policy)
 validate_state(normalized)
 require(normalized==expected,"Source float normalization state")
 return normalized
def check_trace(states,traced,unreferenced):
 validate_state(states);validate_state(traced)
 require(all(states.get(n)==v for n,v in traced.items()) and
  type(unreferenced)is list and unreferenced==sorted(set(states)-set(traced)),
  "Trace state subset of original authenticated state")
 return traced
def direct_divisors(source,state):
 validate_state(state)
 require(type(source)is list and all(type(n)is dict and type(n.get("name"))is str for n in source) and
  len({n["name"]for n in source})==len(source),"Unique public division source inventory")
 roles=_roles(source);nodes={n["name"]:n for n in source};selected={}
 for node in source:
  signature=classification(node["op"],node["target"])
  if signature is None or signature[0]!="divide":continue
  kwargs={}
  for k,v in node["kwargs"]["items"]:
   require(type(k)is str and k not in kwargs,"Public divisor duplicate keyword")
   kwargs[k]=None if v==dict(kind="none")else v
  values=arguments(tuple(node["args"]["items"]),kwargs,signature)
  parameter=values["parameter"]
  if parameter.get("kind")!="fx_value":continue
  base=values["input"]
  require(set(base)=={"kind","name"} and base["kind"]=="fx_value" and roles.get(base["name"])is True,
   "Encrypted tensor dividend required")
  require(set(parameter)=={"kind","name"} and parameter["name"]in nodes,
   "Direct declared public tensor divisor required")
  divisor=nodes[parameter["name"]];name=divisor["target"]
  require(divisor["op"]=="get_attr" and name in state and roles[divisor["name"]]is False,
   "Direct declared public tensor divisor required")
  require(state[name]["dtype"]=="<f8","Public tensor divisor requires exact float64")
  selected[name]=state[name]
 return selected
