"""Directed fixed polynomial fixtures; not Agent generations or new benchmark models."""
from benchmark_suite import Builder, coefficient_tables
from benchmark_graph import digest
from upstream_candidate_helpers import POLYNOMIAL_PROFILE
def cases():
    tables,default=coefficient_tables()
    bank={"Poly_Default":[v if i%2 else 0. for i,v in enumerate(default)]}
    for name,(values,divisor) in zip(("Poly_Tree15First","Poly_Tree15Second","Poly_Tree27"),tables):
        bank[name]=[v/divisor if i%2 else 0. for i,v in enumerate(values)]
    for name,scale in (("Poly_Leaf1",1.),("Poly_Leaf17",1.7),("Poly_Leaf2",2.)):
        bank[name]=[0.,.25/scale,0.,.0625/scale]
    rows=[]
    for name,coeff in bank.items():
        for context in range(3):
            b=Builder([(16,)])
            y=b.node("polynomial",["input0",b.const(coeff)],basis="chebyshev")
            expression=name+"(x)"
            if context==1:
                y=b.node("add",[y,b.node("multiply",["input0",b.const(.125)])])
                expression+=" + x*.125"
            elif context==2:
                y=b.node("rotate",[y],step=1)
                expression+=".rotate(1)"
            model=b.finish(y);model["id"]="fixed_"+name+"_"+str(context)
            source='@hc.func("c,c")\ndef golden(x,zero_ct):\n    return '+expression+'\n'
            rows.append(dict(id=model["id"],model=model,source=source,helper=name,context=context,
                             model_sha256=digest(model),profile=POLYNOMIAL_PROFILE))
    return rows
