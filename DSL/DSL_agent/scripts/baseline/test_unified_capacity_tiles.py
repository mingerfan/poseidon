"""Independent permutation tests, exact old requests and bounded real-helper work."""
import ast,copy,json,math,unittest
import numpy as np
from benchmark_graph import samples,canonical,digest
from benchmark_math import evaluate as math_ref
from benchmark_torch import evaluate as torch_ref
from component_contract import prepare_task,reconstruct_request
from unified_graph_contract import validate_request,validate_candidate
from unified_capacity_constants import packing,PREVIOUS_VERSION
from unified_capacity_tiling import row_stages,template,template_vectors
from unified_capacity_tiled_cases import models
from unified_graph_lowering import candidate_source
from test_unified_capacity_linear import interpreted_output

class CapacityTileTests(unittest.TestCase):
    def test_three_contexts_and_original_models_preserved(self):
        from unified_capacity_diagonal_cases import models as original
        self.assertEqual(canonical(models()[0]),canonical(original()[1]))
        self.assertEqual(canonical(models()[1]),canonical(original()[2]))
        self.assertEqual(len(models()),3)
        self.assertEqual({(g["inputs"][0]["shape"][-1],len(g["constants"]["weight"])) for g in models()},
                         {(3,5),(4,6),(5,7)})

    def test_permutation_bijections_and_dirty_tail(self):
        for width,stride,count in [(3,8,32),(5,8,32),(4,8,32),(6,8,32),(5,8,17),(7,8,3)]:
            p=256;expected=np.arange(count*width)+1.
            value=np.full(p,-999.);value[:count*width]=expected
            for stage in row_stages(width,stride,count,p):
                a,b=np.array(stage["fixed"]),np.array(stage["moving"])
                self.assertFalse(np.any(a*b))
                self.assertEqual(int(a.sum()+b.sum()),count*width)
                value=value*a+np.roll(value*b,-stage["shift"])
            padded=np.zeros(p)
            for batch in range(count):padded[batch*stride:batch*stride+width]=expected[batch*width:(batch+1)*width]
            np.testing.assert_array_equal(value,padded)
            for stage in row_stages(width,stride,count,p,compress=True):
                value=value*np.array(stage["fixed"])+np.roll(value*np.array(stage["moving"]),-stage["shift"])
            np.testing.assert_array_equal(value[:count*width],expected)
            np.testing.assert_array_equal(value[count*width:],np.zeros(p-count*width))

    def test_native_helpers_and_independent_references(self):
        for g in models():
            r=prepare_task(g,{"chunk_period":256});source,strategy=candidate_source(r)
            result=validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
            d=r["constant_origins"]["capacity_tensor_packing"]
            self.assertEqual(d["linear_strategies"],{"n1":"tiled_diagonal"})
            self.assertLessEqual(result["functions"]["golden"]["expanded_cost"],1024)
            self.assertLessEqual(len(list(ast.walk(ast.parse(source)))),4096)
            self.assertLessEqual(len(r["public_constants"]),256)
            self.assertIn('def cap_tile_',source)
            self.assertEqual(strategy["strategy"],"capacity-chunks-v8")
            for inputs in samples(g,16):
                a,b=math_ref(g,inputs),torch_ref(g,inputs)
                for name in a:np.testing.assert_allclose(a[name],b[name],atol=1e-12,rtol=1e-12)
                expected=np.concatenate([a[o["name"]].reshape(-1) for o in g["outputs"]])
                np.testing.assert_allclose(interpreted_output(source,r,inputs),expected,atol=1e-12,rtol=1e-12)

    def test_every_nonzero_weight_has_one_local_contribution(self):
        for g in models():
            weights=g["constants"]["weight"];width=len(weights[0]);outputs=len(weights)
            t=template(weights,17,256);actual={}
            for step,vector in t["weights"].items():
                for dst,weight in enumerate(vector):
                    if not weight:continue
                    src=(dst+step)%256;self.assertNotIn((src,dst),actual);actual[src,dst]=weight
            expected={(batch*t["stride"]+j,batch*t["stride"]+o):weights[o][j]
                      for batch in range(17) for o in range(outputs) for j in range(width) if weights[o][j]}
            self.assertEqual(actual,expected)

    def test_cross_chunk_gather_and_scatter_are_bound(self):
        r=prepare_task(models()[0],{"chunk_period":256});d=r["constant_origins"]["capacity_tensor_packing"]
        blocks=d["linear_tiles"]["n1"]["blocks"]
        self.assertTrue(any(len(b["gather"])==2 for b in blocks))
        self.assertTrue(any(len(b["scatter"])==2 for b in blocks))
        for mutate in (
            lambda d:d["linear_tiles"]["n1"]["blocks"][0].update(batch_offset=1),
            lambda d:d["linear_tiles"]["n1"].update(stride=16),
            lambda d:d["linear_tiles"]["n1"]["blocks"][2]["gather"][0].update(lo=1),
            lambda d:d["bindings"][0].update(phase=99)):
            bad=copy.deepcopy(r);mutate(bad["constant_origins"]["capacity_tensor_packing"])
            bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
            with self.assertRaises(ValueError):validate_request(bad)
        bad=copy.deepcopy(r);key=d["bindings"][0]["key"];bad["public_constants"][key][0]+=0.5
        bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
        with self.assertRaises(ValueError):validate_request(bad)

    def test_v2_requests_reconstruct_exactly_and_keep_original_blocks(self):
        from hecate_python_env import WORK
        count=0
        for f in (WORK/"results/poseidon-capacity-routing-numerical-r398").glob("*/request.json"):
            r=json.loads(f.read_text());self.assertEqual(canonical(reconstruct_request(r)),canonical(r));count+=1
            d=r["constant_origins"].get("capacity_tensor_packing")
            if d:self.assertEqual(d["version"],PREVIOUS_VERSION)
        self.assertEqual(count,8)
        from unified_graph_contract import prepare
        from compiler_configuration import configuration
        c=configuration("poseidon-cpu-eva-w45-v1")
        for g in models()[:2]:
            with self.assertRaisesRegex(ValueError,"public tensor registry budget"):
                prepare(g,c["ckks_config_sha256"],c,chunk_period=256,capacity_packing_version=PREVIOUS_VERSION)

    def test_compact_evidence_request_matches_hash_and_file_gate(self):
        import tempfile
        from pathlib import Path
        from candidate_contract import request_file_bytes,MAX_BYTES
        from benchmark_runner import strict_file
        r=prepare_task(models()[1],{"chunk_period":256})
        self.assertGreater(len(json.dumps(r,indent=2).encode()),MAX_BYTES)
        data=request_file_bytes(r)
        decoded=json.loads(data)
        from unified_graph_contract import validate_request
        validate_request(decoded)
        self.assertEqual(canonical(decoded),canonical(r))
        self.assertLessEqual(len(data),MAX_BYTES)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"request.json";path.write_bytes(data)
            loaded=strict_file(path,MAX_BYTES)
            validate_request(loaded)
            self.assertEqual(loaded,r)
            self.assertEqual(reconstruct_request(strict_file(path,MAX_BYTES)),r)
        with self.assertRaisesRegex(ValueError,"Request size limit"):
            request_file_bytes({"extra":"x"*(MAX_BYTES+1)})
        with self.assertRaises(ValueError):request_file_bytes({"value":float("nan")})

    def test_large_compact_bundle_request_preserves_static_identity(self):
        import tempfile
        from pathlib import Path
        from candidate_contract import request_file_bytes
        from candidate_bundle import check_bundle,sha,implementation
        r=prepare_task(models()[1],{"chunk_period":256});source,_=candidate_source(r)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            (folder/"request.json").write_bytes(request_file_bytes(r))
            (folder/"model.json").write_bytes(canonical(r["model"]))
            (folder/"candidate.py").write_text(source)
            m=dict(format="poseidon-dsl-bundle-v2",
                   files={n:sha(folder/n) for n in ("model.json","request.json","candidate.py")},
                   compiler_configuration=r["compiler_configuration"],
                   origin=dict(request_id=r["request_id"]),agent_generated=False,model_provenance="not_supplied",
                   dependencies=dict(compiler_sha256="a"*64,helper_environment=None,implementation=implementation(),validation_sources={}))
            m["binding"]=digest(m);(folder/"manifest.json").write_bytes(canonical(m))
            observed=check_bundle(folder)
            self.assertEqual(observed["origin"]["request_id"],r["request_id"])
            self.assertNotIn("validation",observed) # Static identity is not numerical proof.
            (folder/"request.json").write_text(json.dumps(r,indent=2))
            m["files"]["request.json"]=sha(folder/"request.json")
            m["binding"]=digest({k:v for k,v in m.items() if k!="binding"})
            (folder/"manifest.json").write_bytes(canonical(m))
            with self.assertRaisesRegex(ValueError,"Bundle file integrity"):check_bundle(folder)

    def test_sparse_and_zero_weight_rows_keep_reference(self):
        g=copy.deepcopy(models()[2]);g["constants"]["weight"][1]=[0.]*5
        g["constants"]["weight"][3]=[0.,0.,-0.125,0.,0.]
        r=prepare_task(g,{"chunk_period":256});source,_=candidate_source(r)
        for inputs in samples(g,16):
            np.testing.assert_allclose(interpreted_output(source,r,inputs),
                                      math_ref(g,inputs)["result"].reshape(-1),atol=1e-12,rtol=1e-12)

if __name__=="__main__":unittest.main()
