"""Independent numerical lowering checks, fixed public bindings and compatibility."""
import ast,copy,json,math,unittest
from pathlib import Path
import numpy as np
from benchmark_graph import samples,canonical,validate
from component_contract import prepare_task,reconstruct_request
from unified_graph_contract import validate_request,validate_candidate
from unified_capacity_constants import packing,key_for
from unified_capacity_linear_cases import models
from unified_graph_lowering import candidate_source
from sandbox_policy import MODULES

def interpreted_output(source,request,inputs):
    """Interpret only checked generated straight-line operations; never exec/import."""
    validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
    p=request["layout"]["input_slot_period"]
    symbols={k:np.asarray(v,dtype=np.float64) for k,v in request["public_constants"].items()}
    symbols["zero_ct"]=np.zeros(p)
    for b in request["layout"]["inputs"]:
        values=np.asarray(inputs[b["name"]]).reshape(-1)[b["offset"]:b["offset"]+b["elements"]]
        symbols[b["dsl_name"]]=np.pad(values,(0,p-len(values)))
    functions={f.name:f for f in ast.parse(source).body}
    def expression(n,local):
        if type(n) is ast.Name:return local[n.id] if n.id in local else symbols[n.id]
        if type(n) is ast.Constant:return n.value
        if type(n) is ast.Subscript:return expression(n.value,local)[expression(n.slice,local)]
        if type(n) is ast.Compare:
            assert len(n.ops)==len(n.comparators)==1 and type(n.ops[0]) is ast.Eq
            return expression(n.left,local)==expression(n.comparators[0],local)
        if type(n) is ast.Call and type(n.func) is ast.Name and n.func.id=="range":
            values=[expression(a,local) for a in n.args]
            assert all(type(v) is int for v in values)
            result=range(*values);assert len(result)<=1024
            return result
        if type(n) is ast.BinOp:
            a,b=expression(n.left,local),expression(n.right,local)
            return {ast.Add:np.add,ast.Sub:np.subtract,ast.Mult:np.multiply,ast.FloorDiv:lambda a,b:a//b,ast.Mod:lambda a,b:a%b}[type(n.op)](a,b)
        if type(n) is ast.UnaryOp and type(n.op) is ast.USub:return -expression(n.operand,local)
        if type(n) is ast.Call and type(n.func) is ast.Attribute and n.func.attr=="rotate":
            assert len(n.args)==1
            step=expression(n.args[0],local);assert type(step) is int
            return np.roll(expression(n.func.value,local),-step)
        if type(n) is ast.Call and type(n.func) is ast.Name and n.func.id in functions:
            return call(functions[n.func.id],[expression(a,local) for a in n.args])
        if type(n) in (ast.List,ast.Tuple):return [expression(x,local) for x in n.elts]
        raise AssertionError("Unexpected emitted operation")
    def call(fn,args):
        assert len(args)==len(fn.args.args)
        local=dict(zip([a.arg for a in fn.args.args],args))
        def statements(body):
            for n in body:
                if type(n) is ast.For:
                    assert type(n.target) is ast.Name and not n.orelse
                    for item in expression(n.iter,local):
                        local[n.target.id]=item;statements(n.body)
                elif type(n) is ast.If:
                    statements(n.body if expression(n.test,local) else n.orelse)
                else:
                    assert type(n) is ast.Assign and len(n.targets)==1 and type(n.targets[0]) is ast.Name
                    local[n.targets[0].id]=expression(n.value,local)
        statements(fn.body[:-1])
        ret=fn.body[-1];assert type(ret) is ast.Return
        return expression(ret.value,local)
    fn=functions["golden"];parts=call(fn,[symbols[a.arg] for a in fn.args.args])
    return np.array([parts[c][j] for c,j in request["layout"]["output_selectors"]])

class CapacityLinearTests(unittest.TestCase):
    def test_independent_math_torch_and_emitted_arithmetic(self):
        from benchmark_math import evaluate as math_ref
        from benchmark_torch import evaluate as torch_ref
        for g in models():
            r=prepare_task(g,{"chunk_period":256});src,_=candidate_source(r)
            for probe in samples(g,16):
                a=math_ref(g,probe);b=torch_ref(g,probe)
                expected=np.concatenate([a[o["name"]].reshape(-1) for o in g["outputs"]])
                actual=interpreted_output(src,r,probe)
                for n in a:np.testing.assert_allclose(a[n],b[n],atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    def test_rank_and_context_count(self):
        from collections import Counter
        self.assertEqual(len(models()),9)
        self.assertEqual(set(Counter(g["id"].rsplit("_",1)[0] for g in models()).values()),{3})
        self.assertEqual({len(g["inputs"][0]["shape"]) for g in models()},{1,2,4})
    def test_public_binding_version_and_reconstruction(self):
        for g in models():
            r=prepare_task(g,{"chunk_period":256})
            from unified_capacity_constants import VERSION
            self.assertEqual(r["constant_origins"]["capacity_tensor_packing"]["version"],VERSION)
            self.assertEqual(canonical(reconstruct_request(r)),canonical(r))
    def test_unique_weights_do_not_consume_one_scalar_entry_each(self):
        g=next(g for g in models() if g["id"]=="capacity_linear_1")
        r=prepare_task(g,{"chunk_period":256})
        self.assertGreater(len({v for row in g["constants"]["weight"] for v in row}),256)
        self.assertLess(len(r["public_constants"]),40)
    def test_tampered_public_vector_rejected(self):
        r=prepare_task(models()[0],{"chunk_period":256})
        key=next(k for k in r["public_constants"] if k.startswith("pv"))
        r["public_constants"][key][0]+=0.125
        # Rehash the public envelope to exercise model-derived binding checks,
        # rather than stopping at the earlier generic request-integrity check.
        from benchmark_graph import digest
        r["request_id"]=digest({k:v for k,v in r.items() if k!="request_id"})
        with self.assertRaisesRegex(ValueError,"Public registry changed"):validate_request(r)
    def test_tampered_source_binding_rejected(self):
        r=prepare_task(models()[0],{"chunk_period":256})
        r["constant_origins"]["capacity_tensor_packing"]["bindings"][0]["chunk"]=3
        with self.assertRaisesRegex(ValueError,"Changed capacity public tensor bindings"):validate_request(r)
    def test_missing_binding_rejected(self):
        r=prepare_task(models()[0],{"chunk_period":256})
        r["constant_origins"].pop("capacity_tensor_packing")
        with self.assertRaisesRegex(ValueError,"Changed capacity public tensor bindings"):validate_request(r)
    def test_masked_weights_exclude_bias_filled_tail(self):
        g=next(g for g in models() if g["id"]=="capacity_linear_0");r=prepare_task(g,{"chunk_period":256})
        packed=packing(g,r["layout"]);node=next(n for n in g["nodes"] if n["op"]=="linear")
        key=key_for(packed,node["id"],"linear_weight",output=0,chunk=1)
        self.assertNotEqual(packed["vectors"][key][0],0.)
        self.assertEqual(packed["vectors"][key][1:],[0.]*255)
    def test_batch_boundary_crosses_ciphertexts(self):
        g=next(g for g in models() if g["id"]=="capacity_linear_2");r=prepare_task(g,{"chunk_period":256})
        packed=packing(g,r["layout"]);node=next(n for n in g["nodes"] if n["op"]=="linear")
        first=packed["vectors"][key_for(packed,node["id"],"linear_weight",output=4,chunk=0)]
        tail=packed["vectors"][key_for(packed,node["id"],"linear_weight",output=4,chunk=1)]
        self.assertEqual(first[:129],[0.]*129);self.assertTrue(any(first[129:]))
        self.assertTrue(any(tail[:2]));self.assertEqual(tail[2:],[0.]*254)
    def test_prior_capacity_requests_unchanged(self):
        from hecate_python_env import WORK
        for batch in ["poseidon-capacity-minimal-r382","poseidon-capacity-remainder-r383"]:
            for folder in (WORK/"results"/batch).iterdir():
                f=folder/"request.json"
                if f.is_file():
                    r=json.loads(f.read_text());self.assertEqual(canonical(reconstruct_request(r)),canonical(r))
    def test_prior_capacity_arithmetic_remains_correct(self):
        from unified_capacity_cases import models as old_models
        from benchmark_math import evaluate
        for g in old_models():
            r=prepare_task(g,{"chunk_period":256});src,_=candidate_source(r)
            for probe in samples(g,4):
                result=evaluate(g,probe)
                expected=np.concatenate([result[o["name"]].reshape(-1) for o in g["outputs"]])
                np.testing.assert_allclose(interpreted_output(src,r,probe),expected,atol=1e-12,rtol=1e-12)
    def test_resource_and_sandbox_limits_remain(self):
        self.assertIn("unified_capacity_constants.py",MODULES)
        g=copy.deepcopy(next(g for g in models() if g["id"]=="capacity_linear_0"))
        g["constants"]["weight"]=[[0.5]*257 for _ in range(64)]
        g["constants"]["bias"]=[0.]*64;validate(g)
        from unified_graph_contract import prepare
        from compiler_configuration import configuration
        config=configuration("poseidon-cpu-eva-w45-v1")
        r=prepare(g,config["ckks_config_sha256"],config,chunk_period=256,capacity_packing_version="capacity-public-tensors-v1")
        with self.assertRaisesRegex(ValueError,"operation budget"):candidate_source(r)

if __name__=="__main__":unittest.main()
