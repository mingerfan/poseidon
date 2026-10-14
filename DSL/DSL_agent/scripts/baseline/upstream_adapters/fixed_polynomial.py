"""Fixed source-locked polynomial calls; no candidate-selected closures or I/O."""
import hashlib
import json
import math
from pathlib import Path
from benchmark_graph import require

# Names identify immutable public functions, never model families.
NAMES = ("Poly_Default", "Poly_Tree15First", "Poly_Tree15Second", "Poly_Tree27",
         "Poly_Leaf1", "Poly_Leaf17", "Poly_Leaf2")

def coefficients(name):
    require(name in NAMES, "Unknown fixed polynomial")
    if name.startswith("Poly_Leaf"):
        scale={"Poly_Leaf1":1., "Poly_Leaf17":1.7, "Poly_Leaf2":2.}[name]
        return [0., .25/scale, 0., .0625/scale]
    from upstream_candidate_helpers import LOCK
    filename="coeffStr.txt" if name=="Poly_Default" else "sgn151527.txt"
    relative="third_party/dacapo/python/poly/poly/data/"+filename
    p=Path("/upstream-poly/poly/data")/filename
    if not p.is_file():
        from workspace_paths import ROOT
        p=ROOT/relative
    raw=p.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==json.loads(LOCK.read_text())["sources"][relative],
            "Changed fixed polynomial coefficients")
    values=[float(v) for v in raw.decode().splitlines() if v.strip()]
    if name=="Poly_Default":
        require(len(values)==96,"Default polynomial size")
        divisor=1.
    else:
        start,stop,divisor={"Poly_Tree15First":(0,16,2.),
                           "Poly_Tree15Second":(16,32,1.7),
                           "Poly_Tree27":(32,60,2.)}[name]
        require(len(values)==60,"Sign polynomial table size")
        values=values[start:stop]
    require(all(math.isfinite(v) for v in values),"Finite coefficients")
    # Actual pinned MPCB only evaluates odd leaf terms; this is explicit metadata.
    return [v/divisor if i%2 else 0. for i,v in enumerate(values)]

def specs():
    return {name:dict(parameters=["c"],result="c",work=256,rotations=[],bootstrap=False,
                     semantics="Pointwise fixed odd-leaf Chebyshev polynomial; not an exact nonlinear function",
                     basis="chebyshev",coefficients=coefficients(name),
                     source="poly.Poly.GenPoly" if name=="Poly_Default" else "poly.MPCB.GenPoly",
                     candidate_coefficients_allowed=False)
            for name in NAMES}

def load(poly,mpcb):
    import numpy as np
    functions={"Poly_Default":poly.GenPoly(), "Poly_Tree15First":poly.poly1,
               "Poly_Tree15Second":poly.poly2,"Poly_Tree27":poly.poly3}
    for name,scale in (("Poly_Leaf1",1.),("Poly_Leaf17",1.7),("Poly_Leaf2",2.)):
        functions[name]=mpcb.GenPoly(["0"],["0",".25","0",".0625"],4,scale=scale)
    def wrap(fn):
        def invoke(value):
            out=fn(np.array([value],dtype=object))
            require(isinstance(out,np.ndarray) and out.shape==(1,) and out.dtype==object,
                    "Fixed polynomial return container")
            return out[0]
        return invoke
    return {name:wrap(fn) for name,fn in functions.items()}

def probe(name,values):
    # Intervention evaluator only. Expected answers use independent graph references.
    coeff=coefficients(name)
    def scalar(x):
        a,b=1.,x
        result=coeff[0]+coeff[1]*x
        for c in coeff[2:]:
            a,b=b,2*x*b-a
            result+=c*b
        return result
    return tuple(scalar(v) for v in values)
