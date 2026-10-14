"""Exact f64 serialization, immutable opt-in identity, and provenance rejection."""
import copy,json,math,re,struct,tempfile,unittest
from pathlib import Path
from compiler_configuration import configuration,validate_configuration,COMPACT_CONFIGURATION,TRACE_SERIALIZATION_POLICY
from poseidon_runtime_compile import compact_f64_trace,prepare_entrypoint,audit_entrypoint,command,ENTRY_POLICY
from test_poseidon_entrypoint import TRACE

def attribute(values):
    raw=b"".join(struct.pack("<d",v) for v in values)
    return 'dense<"0x'+raw.hex().upper()+'"> : tensor<'+str(len(values))+'xf64>'

class TraceSerializationTests(unittest.TestCase):
    def test_exact_values_negative_zero_exponents_and_subnormals(self):
        values=[0.0,-0.0,1.0,-2.0,0.125,1e-5,1e20,math.nextafter(1.,2.),
                math.nextafter(0.,1.),-math.nextafter(0.,1.),float.fromhex("0x1.fffffffffffffp+1023")]
        text="prefix "+attribute(values)+" suffix"
        selected,record=compact_f64_trace(text)
        self.assertTrue(selected.startswith("prefix "));self.assertTrue(selected.endswith(" suffix"))
        if 'dense<[' in selected:
            tokens=re.search(r"dense<\[(.*?)\]>",selected)[1].split(", ")
            self.assertEqual(b"".join(struct.pack("<d",float(v)) for v in tokens),
                             b"".join(struct.pack("<d",v) for v in values))
            self.assertTrue(all("." in v.split("e")[0] for v in tokens))
        self.assertEqual(record["constants"][0]["elements"],len(values))
        self.assertTrue(record["values_bitwise_unchanged"])
    def test_shorter_representation_only_and_other_types_unchanged(self):
        source=attribute([math.nextafter(1.,2.)]*3)
        compact,receipt=compact_f64_trace(source)
        self.assertLessEqual(len(compact),len(source))
        other='value = dense<"0x00000000"> : tensor<1xf32>'
        self.assertEqual(compact_f64_trace(other)[0],other)
        self.assertEqual(compact_f64_trace(other)[1]["constants"],[])
    def test_illegal_buffers_and_nonfinite_rejected(self):
        for source in ['dense<"0x0000"> : tensor<2xf64>',
                       'dense<"0x00"> : tensor<0xf64>',
                       attribute([float("nan")]),attribute([float("inf")]),
                       attribute([-float("inf")])]:
            with self.subTest(source=source),self.assertRaises(ValueError):compact_f64_trace(source)
    def test_raw_and_entry_budgets_both_remain_enforced(self):
        with self.assertRaisesRegex(ValueError,"serialization size"):compact_f64_trace("x"*(16*1024**2+1))
        large=TRACE+"\n// "+attribute([1.0]*(524288+1))
        self.assertGreater(len(large),8*1024**2)
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);(out/"candidate_trace.mlir").write_text(large)
            with self.assertRaisesRegex(ValueError,"entry trace size"):prepare_entrypoint(out,configuration("poseidon-cpu-eva-w45-v1"))
            record=prepare_entrypoint(out,configuration(COMPACT_CONFIGURATION))
            self.assertEqual((out/"candidate_trace.mlir").read_text(),large)
            self.assertLess((out/"candidate_trace.entry.mlir").stat().st_size,8*1024**2)
            self.assertEqual(record["trace_serialization"]["format"],TRACE_SERIALIZATION_POLICY)
            (out/"candidate_trace.mlir").write_text(TRACE+"\n// "+"x"*(8*1024**2))
            with self.assertRaisesRegex(ValueError,"entry trace size"):prepare_entrypoint(out,configuration(COMPACT_CONFIGURATION))
    def test_visibility_additions_must_fit_final_entry_budget(self):
        from poseidon_runtime_compile import entrypoint_trace
        # The helper annotation must not make an admitted source exceed 8 MiB.
        prefix=TRACE+chr(10)+"// "
        source=prefix+"x"*(8*1024**2-len(prefix))
        self.assertEqual(len(source),8*1024**2)
        with self.assertRaisesRegex(ValueError,"selected entry trace size"):entrypoint_trace(source)
    def test_original_compact_selected_and_receipt_are_all_bound(self):
        cfg=configuration(COMPACT_CONFIGURATION);request=dict(compiler_configuration=cfg)
        raw=TRACE+"\n// "+attribute([0.,-.5,1e-5]*8)
        for filename in ["candidate_trace.mlir","candidate_trace.compact.mlir","candidate_trace.entry.mlir","entry-selection.json"]:
            with self.subTest(filename=filename),tempfile.TemporaryDirectory() as t:
                out=Path(t);(out/"candidate_trace.mlir").write_text(raw)
                receipt=prepare_entrypoint(out,cfg);argv=command(45,entry_policy=ENTRY_POLICY)
                audit_entrypoint(out,request,receipt,argv)
                target=out/filename
                if filename.endswith(".json"):
                    changed=json.loads(target.read_text());changed["trace_serialization"]["original_sha256"]="0"*64
                    target.write_text(json.dumps(changed))
                else:target.write_text(target.read_text()+"\n")
                with self.assertRaises(ValueError):audit_entrypoint(out,request,receipt,argv)
    def test_new_configuration_changes_request_identity_preserving_old_versions(self):
        from benchmark_suite import Builder
        from unified_graph_contract import prepare
        from component_contract import reconstruct_request,runner_options
        from run_candidate import parse_args
        b=Builder([(2,)]);g=b.finish(b.node("square",["input0"]))
        configs=[configuration("poseidon-cpu-eva-w45-v1",schema=s) for s in (2,3)]+[configuration(COMPACT_CONFIGURATION)]
        requests=[prepare(g,c["ckks_config_sha256"],c) for c in configs]
        self.assertEqual(len({r["request_id"] for r in requests}),3)
        self.assertEqual(configuration("poseidon-cpu-eva-w45-v1")["schema"],3)
        for cfg,r in zip(configs,requests):
            self.assertEqual(validate_configuration(cfg,cfg["ckks_config_sha256"]),cfg)
            self.assertEqual(reconstruct_request(r),r)
        from application_policy import prepare as application_prepare
        application_request,_=application_prepare(g,"poseidon_cpu","numerical",
                                                   dict(compiler_configuration=COMPACT_CONFIGURATION))
        self.assertEqual(application_request["compiler_configuration"],configs[-1])
        for denied in ("seal-cpu-eva-w45-v1","poseidon-cuda-eva-w45-v1","unregistered"):
            with self.assertRaises(ValueError):
                application_prepare(g,"poseidon_cpu","numerical",dict(compiler_configuration=denied))
        args=parse_args(["--case","example.json","--prepare",*runner_options(requests[-1])])
        self.assertEqual(args.compiler_configuration,COMPACT_CONFIGURATION)
        old,new=configs[1:]
        for key in ("ckks_config_sha256","waterline","parameter_identity","operator_spec_sha256"):
            self.assertEqual(old[key],new[key])
    def test_policy_schema_and_candidate_override_rejected(self):
        cfg=configuration(COMPACT_CONFIGURATION)
        for schema in (True,2,3,5):
            with self.assertRaises(ValueError):configuration(COMPACT_CONFIGURATION,schema=schema)
        with self.assertRaises(ValueError):configuration("poseidon-cpu-eva-w45-v1",schema=4)
        wrong=copy.deepcopy(cfg);wrong["trace_serialization_policy"]="lossy"
        with self.assertRaises(ValueError):validate_configuration(wrong,wrong["ckks_config_sha256"])
if __name__=="__main__":unittest.main()
