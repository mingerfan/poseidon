"""Wide periods require immutable V4 authority; older requests retain their bindings."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from benchmark_graph import canonical,samples,validate
from benchmark_math import evaluate as math_ref
from benchmark_torch import evaluate as torch_ref
from component_contract import prepare_task,reconstruct_request,runner_options
from unified_capacity_layout import ABI,EXTENDED_ABI,WIDE_ABI,WIDE_PERIODS,layout,validate_layout
from unified_graph_contract import validate_request,validate_candidate
from unified_graph_lowering import candidate_source
from packed_input_abi import rotations,PERIODS
from test_unified_capacity_linear import interpreted_output
from poseidon_validation import execution_contract,input_payload,prepare_inputs,execute
from poseidon_artifact_gate import bind_periodic_constants,PRIMES
from unified_capacity_constants import CHANNEL_VERSION
CASES=Path(__file__).parent/"cases/capacity-wide-v1"
def cases():
    return [(json.loads((CASES/row["file"]).read_text()),row["period"])
            for row in json.loads((CASES/"manifest.json").read_text())["cases"]]
def request(g,period):
    return prepare_task(g,dict(chunk_period=period,generation_guidance="explicit-v10"))
class WideCapacityTests(unittest.TestCase):
    def test_all_configured_periods_and_default_remain_explicit(self):
        self.assertEqual(tuple(p for p in WIDE_PERIODS),(512,1024,2048,4096))
        self.assertEqual(PERIODS,(4,8,16,32,64,128,256))
        for g,p in cases():
            value=layout(g,p);self.assertEqual(validate_layout(value),p)
            self.assertEqual(value["execution_abi"],WIDE_ABI)
            self.assertEqual(layout(g)["execution_abi"],EXTENDED_ABI)
    def test_large_period_needs_correct_profile_and_abi(self):
        g,p=cases()[0]
        for profile,abi,period in [("medium-v1",None,p),("resnet20-v1",ABI,p),
                ("resnet20-v1",EXTENDED_ABI,p),("resnet20-v1",WIDE_ABI,256),
                ("resnet20-v1",None,8192)]:
            bad=copy.deepcopy(g);bad["resource_profile"]=profile
            with self.assertRaises(ValueError):layout(bad,period,execution_abi=abi)
    def test_every_binding_authorizes_rotation(self):
        g,p=cases()[-1];r=request(g,p);value=r["layout"]
        self.assertEqual(rotations(p,capacity_layout=value),tuple(1<<i for i in range(12)))
        for wrong in (None,dict(input_slot_period=p,execution_abi=WIDE_ABI)):
            with self.assertRaises(ValueError):rotations(p,capacity_layout=wrong)
        for mutate in [lambda v:v["inputs"][0].update(offset=1),
                lambda v:v["inputs"][0].update(dsl_name="changed"),
                lambda v:v["output_selectors"].__setitem__(4095,[0,0]),
                lambda v:v.update(output_ciphertexts=2)]:
            bad=copy.deepcopy(value);mutate(bad)
            with self.assertRaises(ValueError):rotations(p,capacity_layout=bad)
        with self.assertRaises(ValueError):rotations(2048,capacity_layout=value)
    def test_native_and_flat_checks_use_layout_authority(self):
        g,p=cases()[-1];r=request(g,p)
        source='@hc.func("c,c")\ndef golden(x0,zero_ct):\n    return x0.rotate(2048)\n'
        candidate=dict(schema=1,request_id=r["request_id"],hecate_source=source)
        validate_candidate(candidate,r)
        candidate["hecate_source"]=source.replace("2048","4096")
        with self.assertRaises(ValueError):validate_candidate(candidate,r)
        from decorated_functions import validate as native
        with self.assertRaises(ValueError):
            native(source,{},1,input_names=("x0","zero_ct"),slot_period=p,array_mutation=True)
        from hecate_contract import validate_function,UNIFIED_FLAT_CONTRACT
        validate_function(source,{},1,input_names=("x0","zero_ct"),slot_period=p,
                          contract=UNIFIED_FLAT_CONTRACT,capacity_layout=r["layout"])
    def test_public_construction_uses_same_authority(self):
        from unified_public_contract import CONTRACT
        g,p=cases()[0];r=prepare_task(g,dict(chunk_period=p,construction_profile=CONTRACT))
        source,_=candidate_source(r)
        validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
    def test_v4_packing_version_is_explicit_and_immutable(self):
        from unified_capacity_constants import VERSION
        g,p=cases()[0];r=request(g,p)
        self.assertEqual(r["constant_origins"]["capacity_tensor_packing"]["version"],CHANNEL_VERSION)
        with self.assertRaisesRegex(ValueError,"packing v8"):
            prepare_task(g,dict(chunk_period=p,capacity_packing_version=VERSION))
    def test_rules_and_cli_match_v4(self):
        from run_candidate import parse_args
        g,p=cases()[-1];r=request(g,p)
        self.assertIn(WIDE_ABI,r["rules"]);self.assertNotIn(EXTENDED_ABI,r["rules"])
        self.assertNotIn("At most four model chunks",r["rules"])
        args=parse_args(["--case",str(CASES/(g["id"]+".json")),"--prepare",*runner_options(r)])
        self.assertEqual(args.unified_chunk_period,p);self.assertEqual(args.unified_packing,CHANNEL_VERSION)
    def test_request_codec_and_reconstruction(self):
        from candidate_contract import request_file_bytes
        for g,p in cases():
            r=request(g,p);raw=json.loads(request_file_bytes(r));validate_request(raw)
            self.assertEqual(canonical(raw),canonical(r))
            self.assertEqual(canonical(reconstruct_request(r)),canonical(r))
    def test_two_references_and_generated_arithmetic(self):
        for g,p in cases():
            r=request(g,p);source,_=candidate_source(r)
            for inputs in samples(g,16):
                first,second=math_ref(g,inputs),torch_ref(g,inputs)
                for name in first:np.testing.assert_allclose(first[name],second[name],atol=1e-12,rtol=1e-12)
                expected=np.concatenate([first[o["name"]].reshape(-1) for o in g["outputs"]])
                np.testing.assert_allclose(interpreted_output(source,r,inputs),expected,atol=1e-10,rtol=1e-12)
    def test_slice_axes_steps_and_three_contexts(self):
        for shape,axis,start,stop,step in [([513],0,3,509,3),([3,171],1,166,5,-3),([1,3,171],-1,25,166,3)]:
            g=copy.deepcopy(cases()[0][0]);g["inputs"][0]["shape"]=shape;g["constants"]={}
            g["nodes"]=[dict(id="slice",op="slice",inputs=["image"],outputs=["cut"],
                            attrs=dict(axis=axis,start=start,stop=stop,step=step))]
            g["outputs"]=[dict(name="result",value="cut")]
            r=request(g,512);source,_=candidate_source(r)
            for probe in samples(g,16):
                expected=math_ref(g,probe)["result"].reshape(-1)
                np.testing.assert_allclose(torch_ref(g,probe)["result"].reshape(-1),expected,atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(interpreted_output(source,r,probe),expected,atol=1e-12,rtol=1e-12)
    def test_wide_vector_codec_rejects_unbound_or_changed_layout(self):
        from unified_request_storage import restore_public_constants
        g,p=cases()[0];r=request(g,p)
        for mutate in [lambda x:x["layout"].update(execution_abi=EXTENDED_ABI),
                lambda x:x["layout"]["inputs"][0].update(elements=1),
                lambda x:x["layout"].pop("resource_profile")]:
            bad=copy.deepcopy(r);mutate(bad)
            with self.assertRaises(ValueError):restore_public_constants(bad)
    def test_client_payload_binds_all_ciphertexts_and_decode(self):
        for g,p in cases():
            r=request(g,p);value=r["layout"];n=len(value["inputs"])
            with tempfile.TemporaryDirectory() as t:
                folder=Path(t);data=folder/"data";out=folder/"out";data.mkdir();out.mkdir()
                x=np.full((4,n,p),.25,dtype=np.float64)
                np.savez(data/"arrays.npz",inputs=x)
                prepare_inputs(data,out,r);payload=json.loads((out/"inputs.json").read_text())
                self.assertEqual(set(payload),{"format","execution_abi","input_slot_period","decode_slots",
                                               "input_ciphertexts","output_ciphertexts","samples"})
                self.assertEqual(payload["input_ciphertexts"],n+1)
                self.assertEqual(payload["decode_slots"],p)
                self.assertEqual(np.asarray(payload["samples"]).shape,(4,n+1,p))
                self.assertTrue(np.all(np.asarray(payload["samples"])[:,-1,:]==0))
        legacy=layout(cases()[0][0],256)
        self.assertIsNone(execution_contract(legacy))
        self.assertIsInstance(input_payload(np.zeros((4,4,256)),legacy),list)
    def test_output_protocol_and_full_decode_are_checked(self):
        g,p=cases()[-2];r=request(g,p);c=execution_contract(r["layout"])
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)
            def report(bad=False):
                return dict(backend="poseidon_cpu",encrypted_execution=True,bootstrap=False,
                    security="tc128",q=list(PRIMES[:13]),p=list(PRIMES[13:]),executor_secret_key=False,
                    format="poseidon-agent-execution-v2",execution_abi=WIDE_ABI,input_slot_period=p,
                    decoded_slots=256 if bad else p,input_ciphertexts=c["input_ciphertexts"],
                    output_ciphertexts=c["output_ciphertexts"],
                    outputs=np.tile(np.arange(p),(4,c["output_ciphertexts"],1)).tolist())
            def run(*args):
                (out/"execution.json").write_text(json.dumps(report()));return 0
            execute({},out,r,run,None)
            result=np.load(out/"decrypted.npy")
            self.assertEqual(result.shape,(4,2049));self.assertEqual(result[0,2047],2047)
            def bad_run(*args):
                (out/"execution.json").write_text(json.dumps(report(True)));return 0
            with self.assertRaisesRegex(ValueError,"protocol identity"):execute({},out,r,bad_run,None)
    def test_periodic_bundle_needs_full_authority(self):
        g,p=cases()[-1];r=request(g,p)
        with tempfile.TemporaryDirectory() as t:
            f=Path(t);plan=f/"raw.json"
            raw=dict(initialization=[dict(kind="encode",output=1,payload=dict(kind="inline",values=[.125]))],execution=[])
            plan.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):bind_periodic_constants(plan,f,f/"unbound",p)
            record=bind_periodic_constants(plan,f,f/"bound",p,capacity_layout=r["layout"])
            self.assertEqual(record["format"],"poseidon-periodic-binding-v2")
            self.assertEqual(record["execution_abi"],WIDE_ABI)
            blob=next((f/"bound/bundle/data").glob("*.bin"))
            self.assertEqual(blob.stat().st_size,16384*8)
            old=bind_periodic_constants(plan,f,f/"old",256)
            self.assertEqual(old["format"],"poseidon-periodic-binding-v1")
            self.assertNotIn("execution_abi",old)
    def test_successive_rotations_preserve_values_and_primitive_budget(self):
        from unified_capacity_lowering import accumulate_rotated
        from unified_graph_lowering import Emitter,Scalar
        g,p=cases()[-1];r=request(g,p)
        for offsets in ([16,32,64,128,256,512],[1008,2032,3056,4080],[0,1008,1008,2032]):
            em=Emitter(r);em.capacity_inline_rotations=True
            terms=[(offset,em.emit("x0 * "+str((i+1)/8))) for i,offset in enumerate(offsets)]
            result=accumulate_rotated(em,terms)
            source='@hc.func("c,c")\ndef golden(x0,zero_ct):\n'+"\n".join(em.lines)+"\n    return ["+result.name+"]\n"
            validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
            for probe in samples(g,4):
                x=probe["image"].reshape(-1)
                expected=sum(np.roll(x,-offset)*((i+1)/8) for i,offset in enumerate(offsets))
                np.testing.assert_allclose(interpreted_output(source,r,probe),expected,atol=1e-12,rtol=1e-12)
    def test_original_full_size_stem_reference_and_candidate(self):
        import ast,hashlib
        path=Path(__file__).parent/"fixtures/wide-capacity-compatibility-v1/stem-model.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "3b3a5ab74bf5a2aa49b675550b59173fbdd1f7f2bc3ca415a1269aaa14a9cb58")
        g=json.loads(path.read_text());r=request(g,4096);source,_=candidate_source(r)
        self.assertLessEqual(len(list(ast.walk(ast.parse(source)))),4096)
        check=validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
        self.assertLessEqual(check["functions"]["golden"]["expanded_cost"],1024)
        for probe in samples(g,4):
            a,b=math_ref(g,probe),torch_ref(g,probe)
            np.testing.assert_allclose(a["channel_means"],b["channel_means"],atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(interpreted_output(source,r,probe),a["channel_means"].reshape(-1),atol=1e-12,rtol=1e-12)
    def test_previous_packages_reconstruct_requests_and_sources(self):
        import hashlib
        folder=Path(__file__).parent/"fixtures/wide-capacity-compatibility-v1"
        manifest=json.loads((folder/"manifest.json").read_text())
        self.assertEqual(manifest["format"],"wide-capacity-compatibility-fixtures-v1")
        self.assertEqual(len(manifest["requests"]),5)
        self.assertEqual(set(p.name for p in (folder/"requests").iterdir()),
                         set(row["name"] for row in manifest["requests"]))
        count=0
        for row in manifest["requests"]:
            f=folder/"requests"/row["name"]/"request.json"
            candidate=f.parent/"candidate.py"
            self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),row["request_sha256"])
            self.assertEqual(hashlib.sha256(candidate.read_bytes()).hexdigest(),row["candidate_sha256"])
            raw=json.loads(f.read_text());validate_request(raw)
            self.assertEqual(canonical(reconstruct_request(raw)),canonical(raw))
            source,_=candidate_source(raw)
            self.assertEqual(source,candidate.read_text());count+=1
        self.assertEqual(count,5)
if __name__=="__main__":unittest.main()
