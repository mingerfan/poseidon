"""RuntimePlan migration boundaries; static tests do not count as FHE execution."""
import copy,json,tempfile,unittest,contextlib,io
from pathlib import Path
from benchmark_suite import Builder
from component_contract import prepare_task
from poseidon_artifact_gate import inspect,canonical,sha,bind_periodic_constants
from poseidon_runtime_paths import SPEC
from compiler_configuration import configuration,PROFILE_SHA256

class RuntimeGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        b=Builder([(2,)]);self.request=prepare_task(b.finish(b.node("square",["input0"])))
        self.spec=json.loads(SPEC.read_text())
        self.plan=dict(format_version=1,plan_id="1",target=dict(capability_version=1,device_counts=[0],
            operator_spec=dict(id="poseidon-agent-cpu-v1",version=1,source_sha256=sha(SPEC.read_bytes())),
            target_id="poseidon-ckks-cpu",world_size=1),
            values=[self.value("0"),self.value("z"),self.value("1")],
            external_inputs=["0","z"],final_outputs=["1"],initialization=[],finalization=[],
            execution=[dict(kind="compute",ordinal=0,op="negate",inputs=["0"],output="1",place=dict(kind="host",rank=0))])
    def tearDown(self):self.tmp.cleanup()
    def value(self,id,**kw):
        return dict(dict(id=id,kind="ciphertext",level=12,scale_log2=45,components=2,
                    context="poseidon-agent-n15-q13-p1-tc128-v1",ntt=True,place=dict(kind="host",rank=0)),**kw)
    def run_gate(self):
        p=self.root/"plan.json";p.write_bytes(canonical(self.plan))
        return inspect(p,SPEC,self.request,bundle=self.root/"bundle")
    def binary(self,op,scale):
        self.plan["values"].append(self.value("p",kind="plaintext",components=1,scale_log2=scale))
        self.plan["initialization"]=[dict(kind="encode",ordinal=1,output="p",payload=dict(kind="inline",values=[1.]))]
        self.plan["execution"][0].update(op=op,inputs=["0","p"])
    def test_valid(self):self.assertEqual(self.run_gate()["output_count"],1)
    def test_scale_over_180_allowed_by_actual_capacity(self):
        self.binary("mul_cp",250);self.plan["values"][2]["scale_log2"]=295
        self.assertEqual(self.run_gate()["instruction_count"],2)
    def test_insufficient_capacity_rejected(self):
        self.plan["values"][2].update(level=0,scale_log2=75)
        with self.assertRaisesRegex(ValueError,"capacity"):self.run_gate()
    def test_intermediate_multiply_overflow(self):
        self.binary("mul_cp",760);self.plan["values"][2]["scale_log2"]=805
        with self.assertRaisesRegex(ValueError,"capacity"):self.run_gate()
    def test_wrong_output_scale(self):
        self.plan["values"][2]["scale_log2"]=46
        with self.assertRaisesRegex(ValueError,"propagation"):self.run_gate()
    def test_add_scale_mismatch(self):
        self.binary("add_cp",44)
        with self.assertRaisesRegex(ValueError,"scale mismatch"):self.run_gate()
    def test_unrequested_rotation(self):
        self.plan["execution"][0].update(op="rotate",attrs=dict(steps=7))
        with self.assertRaisesRegex(ValueError,"rotation"):self.run_gate()
    def test_bootstrap_blocked(self):
        self.plan["execution"][0]["op"]="boot"
        with self.assertRaisesRegex(ValueError,"capability"):self.run_gate()
    def test_plaintext_type_confusion(self):
        self.binary("mul_cc",45);self.plan["values"][2]["scale_log2"]=90
        with self.assertRaisesRegex(ValueError,"type/level"):self.run_gate()
    def test_duplicate_value(self):
        self.plan["values"].append(copy.deepcopy(self.plan["values"][0]))
        with self.assertRaisesRegex(ValueError,"Duplicate"):self.run_gate()
    def test_undefined_input(self):
        self.plan["execution"][0]["inputs"]=["unknown"]
        with self.assertRaisesRegex(ValueError,"undefined"):self.run_gate()
    def test_chain_exhaustion(self):
        self.plan["execution"][0].update(op="rescale",attrs=dict(target_level=-1,target_scale_log2=0))
        with self.assertRaisesRegex(ValueError,"exhaustion"):self.run_gate()
    def test_modswitch_preserves_scale(self):
        self.plan["execution"][0].update(op="mod_switch",attrs=dict(target_level=0))
        self.plan["values"][2]["level"]=0
        self.run_gate()
        self.plan["values"][2]["scale_log2"]=40
        with self.assertRaisesRegex(ValueError,"propagation"):self.run_gate()
    def test_rescale_override_rejected(self):
        self.binary("mul_cp",45);self.plan["values"][2]["scale_log2"]=90
        self.plan["values"].append(self.value("out",level=11,scale_log2=30))
        self.plan["execution"].append(dict(kind="compute",ordinal=2,op="rescale",inputs=["1"],output="out",
            attrs=dict(target_level=11,target_scale_log2=30),place=dict(kind="host",rank=0)))
        self.plan["final_outputs"]=["out"];self.run_gate()
        self.plan["execution"][-1]["attrs"]["target_scale_log2"]=31
        with self.assertRaisesRegex(ValueError,"alter interpretation"):self.run_gate()
    def test_bundle_hash_and_periodic_semantics(self):
        self.binary("mul_cp",45);self.plan["values"][2]["scale_log2"]=90
        raw=canonical(self.plan);p=self.root/"plan.json";p.write_bytes(raw)
        bind_periodic_constants(p,self.root/"unused",self.root/"bound",4)
        self.assertEqual(p.read_bytes(),raw)
        inspect(self.root/"bound/plan.json",SPEC,self.request,bundle=self.root/"bound/bundle")
        blob=next((self.root/"bound/bundle/data").glob("*.bin"));blob.write_bytes(b"x"*131072)
        with self.assertRaisesRegex(ValueError,"hash"):inspect(self.root/"bound/plan.json",SPEC,self.request,bundle=self.root/"bound/bundle")
    def test_constant_free_binding_omits_bundle_reference(self):
        raw=canonical(self.plan);p=self.root/"plan.json";p.write_bytes(raw)
        record=bind_periodic_constants(p,self.root/"unused",self.root/"bound",4)
        bound=json.loads((self.root/"bound/plan.json").read_text())
        self.assertNotIn("plaintext_bundle",bound)
        self.assertEqual(record["mapping"],[])
        self.assertEqual(p.read_bytes(),raw)
        self.assertEqual(json.loads((self.root/"bound/bundle/manifest.json").read_text())["blobs"],[])
        self.assertEqual(inspect(self.root/"bound/plan.json",SPEC,self.request,
                         bundle=self.root/"bound/bundle")["instruction_count"],1)
    def test_bundle_reference_without_encode_rejected(self):
        self.plan["plaintext_bundle"]=dict(id="unused",version=1,manifest_sha256="sha256:"+"0"*64)
        with self.assertRaisesRegex(ValueError,"exactly when"):self.run_gate()
    def test_bundle_payload_without_reference_rejected(self):
        self.binary("mul_cp",45);self.plan["values"][2]["scale_log2"]=90
        self.plan["initialization"][0]["payload"]=dict(kind="bundle",content="sha256:"+"0"*64)
        with self.assertRaisesRegex(ValueError,"exactly when"):self.run_gate()
    def test_inline_payload_with_bundle_reference_rejected(self):
        self.binary("mul_cp",45);self.plan["values"][2]["scale_log2"]=90
        self.plan["plaintext_bundle"]=dict(id="unused",version=1,manifest_sha256="sha256:"+"0"*64)
        with self.assertRaisesRegex(ValueError,"exactly when"):self.run_gate()
    def test_spec_digest_mismatch(self):
        self.plan["target"]["operator_spec"]["source_sha256"]="sha256:"+"0"*64
        with self.assertRaisesRegex(ValueError,"target"):self.run_gate()
    def test_legacy_binding_is_stable_and_distinct(self):
        old=configuration("seal-cpu-eva-w45-v1");new=configuration("poseidon-cpu-eva-w45-v1")
        self.assertEqual(old["schema"],1);self.assertEqual(old["ckks_config_sha256"],PROFILE_SHA256)
        self.assertNotEqual(old["identity_sha256"],new["identity_sha256"])
    def test_no_seal_cli_execution_choice(self):
        from run_candidate import parse_args
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):parse_args(["--case","x.json","--self-test","--compiler-configuration","seal-cpu-eva-w45-v1"])
if __name__=="__main__":unittest.main()
