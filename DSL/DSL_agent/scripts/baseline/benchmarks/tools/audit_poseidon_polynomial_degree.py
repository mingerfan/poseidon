"""Exact leading-degree evidence for immutable polynomial graphs; not an FHE test."""
import json,math,hashlib,sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import require,digest,canonical,validate
from workspace_paths import WORK

@dataclass(frozen=True)
class Leading:
    degree: object
    coefficient: object

def constant(value):return Leading(0,Fraction(float(value)))
def operation(a,b,op):
    if op=="multiply":
        if a==constant(0) or b==constant(0):return constant(0)
        if a.degree is None or b.degree is None:return Leading(None,None)
        if a.coefficient is None or b.coefficient is None:return Leading(a.degree+b.degree,None)
        return Leading(a.degree+b.degree,a.coefficient*b.coefficient)
    if a.degree is None or b.degree is None:return Leading(None,None)
    if a.degree!=b.degree:
        return a if a.degree>b.degree else Leading(b.degree,-b.coefficient if op=="subtract" and b.coefficient is not None else b.coefficient)
    if a.coefficient is None or b.coefficient is None:return Leading(a.degree,None)
    c=a.coefficient+b.coefficient if op=="add" else a.coefficient-b.coefficient
    if c or a.degree==0:return Leading(a.degree,c)
    return Leading(None,None)  # Leading cancellation does not prove all lower terms cancel.

def polynomial(x,coeff):
    degree=max((i for i,c in enumerate(coeff) if c!=0),default=0)
    if x.degree is None:return Leading(None,None)
    if x.degree==0:return Leading(0,None)
    if x.coefficient is None:return Leading(degree*x.degree,None)
    factor=Fraction(float(coeff[degree]))*(2**(degree-1) if degree else 1)
    return Leading(x.degree*degree,factor*x.coefficient**degree)

def array_of(values):
    values=np.asarray(values,dtype=np.float64);out=np.empty(values.shape,dtype=object)
    for i,v in enumerate(values.flat):out.flat[i]=constant(v)
    return out

def restricted(model):
    checked=validate(model);values={k:array_of(v) for k,v in model["constants"].items()}
    for j,spec in enumerate(model["inputs"]):
        arr=array_of(np.zeros(spec["shape"],dtype=np.float64))
        if j==0:arr.flat[0]=Leading(1,Fraction(1))
        values[spec["name"]]=arr
    for node in model["nodes"]:
        op=node["op"];xs=[values[k] for k in node["inputs"]];attrs=node["attrs"]
        if op in ("add","subtract","multiply"):
            a,b=np.broadcast_arrays(xs[0],xs[1]);y=np.empty(a.shape,dtype=object)
            for i,(x,z) in enumerate(zip(a.flat,b.flat)):y.flat[i]=operation(x,z,op)
        elif op=="negate":
            y=np.empty(xs[0].shape,dtype=object)
            for i,x in enumerate(xs[0].flat):y.flat[i]=Leading(x.degree,-x.coefficient if x.coefficient is not None else None)
        elif op=="slice":
            indexes=[slice(None)]*xs[0].ndim
            indexes[attrs["axis"]]=slice(attrs["start"],attrs["stop"],attrs["step"])
            y=xs[0][tuple(indexes)].copy()
        elif op=="concat":y=np.concatenate(xs,axis=attrs["axis"])
        elif op=="polynomial":
            require(attrs["basis"]=="chebyshev","Exact Chebyshev basis required")
            coeff=np.asarray(model["constants"][node["inputs"][1]],dtype=np.float64)
            y=np.empty(xs[0].shape,dtype=object)
            for i,x in enumerate(xs[0].flat):y.flat[i]=polynomial(x,coeff)
        else:raise ValueError("Degree audit does not implement "+op)
        require(list(y.shape)==checked["shapes"][node["outputs"][0]],"Shape identity")
        values[node["outputs"][0]]=y
    return [values[o["value"]].flat[0] for o in model["outputs"]]

def self_test():
    x=Leading(1,Fraction(1))
    require(polynomial(x,[1,0,2])==Leading(2,Fraction(4)),"Chebyshev leading coefficient")
    require(operation(x,constant(0),"multiply")==constant(0),"Zero multiplication")
    require(operation(Leading(2,Fraction(1)),Leading(2,Fraction(-1)),"add").degree is None,"Cancellation must be unproven")
    require(operation(Leading(2,Fraction(4)),x,"subtract")==Leading(2,Fraction(4)),"Lower-degree subtraction")
    require(operation(constant(1),x,"subtract")==Leading(1,Fraction(-1)),"Signed leading coefficient")
    return 5

def main():
    tests=self_test();prior=WORK/"results/poseidon-math-native-r269";plan=json.loads((prior/"plan.json").read_text())
    out=WORK/"results/poseidon-polynomial-degree-r311";require(not out.exists(),"Fresh symbolic evidence");out.mkdir();rows=[]
    for item in plan["models"]:
        model=json.loads((prior/item["model_id"]/"model.json").read_text());require(digest(model)==item["model_sha256"],"Original model hash")
        values=restricted(model);lead=values[0]
        require(lead.degree is not None and lead.coefficient is not None and lead.coefficient!=0,"Nonzero exact leading term must be proven")
        stages=[max(i for i,c in enumerate(model["constants"][n["inputs"][1]]) if c!=0) for n in model["nodes"] if n["op"]=="polynomial"]
        rows.append(dict(model_id=model["id"],model_sha256=digest(model),status="degree_proven_on_symbolic_restriction",
                         polynomial_degrees=stages,output_degree=lead.degree,
                         minimum_ciphertext_multiplication_depth=(lead.degree-1).bit_length(),
                         leading_coefficient_nonzero=True,leading_numerator_bits=abs(lead.coefficient.numerator).bit_length(),
                         leading_denominator_bits=lead.coefficient.denominator.bit_length()))
    report=dict(format="poseidon-polynomial-symbolic-degree-v1",planned=12,rows=rows,unit_tests=dict(passed=tests,failed=0),
                restriction="First logical input element is indeterminate t; every other logical input element is zero",
                assumptions=["Exact declared polynomial arithmetic","Addition, public scaling and rotation do not increase degree",
                             "Ciphertext multiplication multiplies two polynomial messages","No bootstrap or alternative approximation"],
                caveats=["Not an encrypted execution","Not proof of impossibility for all CKKS schedules or finite-test candidates",
                         "Does not justify changing scale, model, reference, compiler or security parameters"],
                actual_compile=False,encrypted_execution=False,agent_calls=0,paid_calls=0,
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/"report.json").write_bytes(canonical(report))
    print(json.dumps(dict(models=len(rows),degrees=sorted(set(x["output_degree"] for x in rows)),
                         depth_lower_bounds=sorted(set(x["minimum_ciphertext_multiplication_depth"] for x in rows)),unit_tests=tests)),flush=True)
if __name__=="__main__":main()
