"""Bounded JSON reading and atomic report writing shared by application and tools."""
import json

def dump(path,value):
    temporary=path.with_name(path.name+".partial")
    temporary.write_bytes(json.dumps(value,indent=2,sort_keys=True,allow_nan=False).encode()+b"\n")
    temporary.replace(path)

def strict_file(path,limit):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>limit: raise ValueError("Benchmark file gate")
    def pairs(items):
        result={}
        for k,v in items:
            if k in result: raise ValueError("Duplicate JSON key")
            result[k]=v
        return result
    def nonfinite(x): raise ValueError("Nonfinite JSON")
    return json.loads(path.read_bytes(),object_pairs_hook=pairs,parse_constant=nonfinite)
