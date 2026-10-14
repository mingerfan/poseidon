"""Original rectangular cases plus an independent third tile context."""
import copy
from benchmark_graph import validate
from unified_capacity_diagonal_cases import models as original_models

def models():
    original=original_models()
    result=[original[1],original[2]]
    g=copy.deepcopy(original[1]);g["id"]="capacity_tiled_linear_2"
    g["inputs"][0]["shape"]=[1,1,65,5]
    g["constants"]["weight"]=[[(i*5+j+1)/128. for j in range(5)] for i in range(7)]
    g["constants"]["bias"]=[(i-2)/256. for i in range(7)]
    validate(g);result.append(g)
    return result
