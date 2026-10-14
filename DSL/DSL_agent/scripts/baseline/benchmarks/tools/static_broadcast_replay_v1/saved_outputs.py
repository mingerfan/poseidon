"""Restore declaration order after canonical JSON key sorting."""
from benchmark_graph import require
def declared_outputs(values,names):
 require(type(values)is dict and type(names)is list and len(names)==len(set(names))
  and set(values)==set(names),"saved_output_inventory")
 return {name:values[name]for name in names}
