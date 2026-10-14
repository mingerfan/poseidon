"""Chunk packing/layout and independent small plaintext checks; not FHE."""
import copy,json,math,tempfile,unittest
from pathlib import Path
import numpy as np
from benchmark_graph import digest,samples,signature
from benchmark_math import evaluate
from benchmark_torch import evaluate as torch_reference
from benchmark_suite import Builder
from unified_graph_contract import prepare,validate_request,validate_candidate
from unified_graph_lowering import lower
from unified_chunk_layout import ABI,layout
from unified_chunk_cases import cases,source_for
from compiler_configuration import PROFILE_SHA256,configuration
from test_unified_graph import Vec

def request(row):
    return prepare(row["model"],PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"),chunk_period=row["period"],
                   construction_profile=row.get("profile"),construction=row.get("exercise"))

def probe(r,source,inputs):
    from candidate_trace import evaluate_tree
    from unified_public_contract import normalize
    constants=r["public_constants"]
    if "construction_profile" in r:
        value=normalize(source,r);source=value["source"];constants=value["constants"]
    p=r["layout"]["input_slot_period"];args={}
    for b in r["layout"]["inputs"]:
        flat=inputs[b["name"]].reshape(-1)
        part=flat[b["offset"]:b["offset"]+b["elements"]]
        args[b["dsl_name"]]=Vec(np.pad(part,(0,p-len(part))))
    args["zero_ct"]=Vec(np.zeros(p))
    constants={k:Vec(v if type(v) is list else [v]*p) for k,v in constants.items()}
    out=evaluate_tree(source,constants,encrypted_inputs=args)
    return np.asarray([out[i].v[j] for i,j in r["layout"]["output_selectors"]])

class ChunkTests(unittest.TestCase):
    def test_mathematical_torch_and_rule_baseline_probes(self):
        for row in cases():
            with self.subTest(case=row["name"]):
                r=request(row);source=source_for(row,r)
                validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
                for inputs in samples(row["model"],16):
                    a=evaluate(row["model"],inputs);b=torch_reference(row["model"],inputs)
                    expected=np.concatenate([a[o["name"]].reshape(-1) for o in row["model"]["outputs"]])
                    for name in a:np.testing.assert_allclose(a[name],b[name],atol=1e-12,rtol=1e-12)
                    # evaluate_tree handles flat DSL only. Native helper variants are
                    # contract/intervention-checked above and tested in real FHE separately.
                    numeric_source=lower(r) if row.get("exercise") else source
                    np.testing.assert_allclose(probe(r,numeric_source,inputs),expected,atol=1e-12,rtol=1e-12)
    def test_input_arrays_are_real_chunks_and_zero_tail(self):
        from unified_graph_prepare import prepare_case
        row=cases()[0];r=request(row)
        with tempfile.TemporaryDirectory() as folder:
            prepare_case(row["model"],Path(folder),row["period"])
            with np.load(Path(folder)/"arrays.npz") as data:
                self.assertEqual(data["inputs"].shape,(4,3,4))
                self.assertTrue(np.all(data["inputs"][:,-1,1:]==0))
                for batch,original in zip(data["inputs"],samples(row["model"],4)):
                    np.testing.assert_array_equal(batch.reshape(-1)[:9],original["input0"])
    def test_each_layout_field_is_bound_even_after_rehash(self):
        for mutation in ("offset","elements","dsl_name","selectors","output_chunk"):
            r=request(cases()[0])
            if mutation in ("offset","elements"):r["layout"]["inputs"][1][mutation]+=1
            elif mutation=="dsl_name":r["layout"]["inputs"][1]["dsl_name"]="x"
            elif mutation=="selectors":r["layout"]["output_selectors"][4]=[0,0]
            else:r["layout"]["outputs"][0]["chunks"][1]["ciphertext"]=0
            r["request_id"]=digest({k:v for k,v in r.items() if k!="request_id"})
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,"Changed chunk physical layout"):
                validate_request(r)
    def test_existing_ciphertext_budgets_not_increased(self):
        b=Builder([(17,)]);g=b.finish(b.node("negate",["input0"]))
        with self.assertRaisesRegex(ValueError,"input ciphertext budget"):layout(g,4)
        b=Builder([(1,)]);g=b.finish(b.node("linear",["input0",b.const([[1.]]*17)]))
        with self.assertRaisesRegex(ValueError,"output ciphertext budget"):layout(g,4)
        for p in (True,3,512):
            with self.assertRaises(ValueError):layout(cases()[0]["model"],p)
    def test_default_request_and_layout_stay_legacy(self):
        g=cases()[0]["model"]
        old=prepare(g,PROFILE_SHA256);new=prepare(g,PROFILE_SHA256,chunk_period=4)
        self.assertEqual(old["layout"]["execution_abi"],"unified-periodic-inputs-v1")
        self.assertEqual(old["layout"]["input_slot_period"],16)
        self.assertEqual(new["layout"]["execution_abi"],ABI)
        self.assertNotEqual(old["request_id"],new["request_id"])
        forged=copy.deepcopy(new);forged["rules"]=old["rules"]
        forged["request_id"]=digest({k:v for k,v in forged.items() if k!="request_id"})
        with self.assertRaisesRegex(ValueError,"request contract"):validate_request(forged)
    def test_helper_requires_verified_chunk_binding(self):
        with self.assertRaisesRegex(ValueError,"separately verified"):
            prepare(cases()[0]["model"],PROFILE_SHA256,chunk_period=4,helper_profile="upstream-poly-silu-v1")
    def test_shared_driver_keeps_logical_tensor_count_distinct(self):
        from cipher_abi import execution_options
        opt=execution_options(request(cases()[0])["layout"])
        self.assertEqual(opt["expected_inputs"],4)
        self.assertEqual(opt["logical_inputs"],3)
        self.assertEqual(opt["logical_tensor_count"],1)
    def test_wrong_whole_tensor_rotation_is_detected(self):
        row=cases()[1];r=request(row)
        wrong='@hc.func("c,c,c,c")\ndef golden(x,y,z,zero_ct):\n return [x.rotate(2),y.rotate(2),z.rotate(2)]\n'
        inputs=samples(row["model"],4)[1]
        expected=evaluate(row["model"],inputs)["output0"]
        self.assertFalse(np.allclose(probe(r,wrong,inputs),expected))
    def test_public_profile_is_also_chunk_bound(self):
        row=cases()[0];row["profile"]="hecate-unified-public-v1";r=request(row);source=lower(r)
        validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
        inputs=samples(row["model"],4)[1]
        np.testing.assert_allclose(probe(r,source,inputs),evaluate(row["model"],inputs)["output0"],atol=1e-12)

if __name__=="__main__":unittest.main()
