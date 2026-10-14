import copy, hashlib, json, os, subprocess, sys, unittest
from pathlib import Path
import numpy as np
from benchmark_graph import digest,canonical
from pytorch_ordered_contract import (
    FORMAT,PACKAGE_FORMAT,prepare,source_identity,capture_binding,check_reference,
    check_capture_metadata,check_digest,validate_package,
)
from pytorch_weight_package import read,binding
import trusted_pytorch_import as legacy

PROFILE="general-tensors-v1"
MEASUREMENTS=[]
CACHE={}

def captured(name):
    if name in CACHE:return CACHE[name]
    package=Path("/fixtures")/name
    spec,_=prepare(package/"manifest.json")
    arrays,_=read(package,spec["weights"],resource_profile=PROFILE)
    expected=binding(arrays,spec["weights"])
    out=Path("/out")/("worker-"+name)
    out.mkdir()
    request=out/"request.json"
    request.write_text(json.dumps(dict(manifest=spec,resource_profile=PROFILE,
        runtime=legacy.runtime_identity(),source_hashes=source_identity())))
    env={k:v for k,v in os.environ.items() if k in (
        "LD_LIBRARY_PATH","POSEIDON_PLATFORM","POSEIDON_WORK_ROOT","HOME",
        "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","LANG")}
    env.update(PYTHONPATH="/app:"+str(package),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
    code="from pytorch_ordered_intake import capture_request;capture_request(*__import__('sys').argv[1:])"
    p=subprocess.run([sys.executable,"-B","-c",code,str(request),str(package),str(out)],
                     env=env,text=True,capture_output=True,timeout=25)
    (out/"worker.log").write_text(p.stdout+p.stderr)
    if p.returncode:raise AssertionError("Worker failed: "+p.stderr[-6000:])
    high=json.loads((out/"original.json").read_text())
    report=json.loads((out/"capture-reference.json").read_text())
    low,bound=capture_binding(high,spec,resource_profile=PROFILE,
        capture_metadata=report["capture_metadata"],expected_weight_binding=expected)
    check_reference(report,high,low,spec,resource_profile=PROFILE,expected_weight_binding=expected)
    measurement=dict(case=name,groups=report["probes"],
        compared_values=report["finite_reference"]["compared_values"],
        reference=report["finite_reference"],source_hashes=report["source_hashes"],
        original_sha256=digest(high),model_sha256=digest(low),
        ordered_binding=bound["binding_sha256"],source_package_sha256=digest(spec))
    MEASUREMENTS.append(measurement)
    CACHE[name]=(high,low,bound,report,spec,expected)
    return CACHE[name]

class SourceWorkerTests(unittest.TestCase):
    def test_six_categories_with_loaded_weights_and_real_source_workers(self):
        for name in ("linear","conv","residual","norm","matrix","transpose_weight"):
            with self.subTest(name=name):
                high,low,bound,report,spec,expected=captured(name)
                self.assertEqual(report["finite_reference"]["groups"],16)
                self.assertTrue(report["finite_reference"]["input_storage_isolated"])
                self.assertFalse(report["encrypted_execution"])
                self.assertEqual(high["format"],"poseidon-operator-graph-v2")
                self.assertEqual(bound["capture_policy_version"],FORMAT)
                self.assertEqual(report["capture_metadata"]["weight_binding"],expected)

    def test_public_split_view_has_original_weight_and_indices(self):
        high,low,bound,report,spec,expected=captured("split_weight")
        alias=report["capture_metadata"]["ordered_alias_capture"]
        self.assertTrue(alias["capture_public_constant_views"])
        for name,view in alias["capture_public_constant_views"].items():
            self.assertIn(view["source"],alias["capture_emission_state_constant_sources"])
            source=np.asarray(alias["capture_emissions"]["constants"][view["source"]])
            np.testing.assert_array_equal(np.asarray(alias["capture_emissions"]["constants"][name]),
                                          source.ravel()[np.asarray(view["indices"])])

    def test_static_split_ports_preserve_ordered_aliases(self):
        high,low,bound,report,spec,expected=captured("split")
        self.assertTrue(any(row.get("rule")=="mutation_ssa"
            for row in report["capture_metadata"]["ordered_alias_capture"]["normalization"]["events"]))

    def test_broadcast_copy_reference_uses_separate_input_storage(self):
        high,low,bound,report,spec,expected=captured("broadcast_copy")
        self.assertTrue(report["finite_reference"]["input_storage_isolated"])
        self.assertTrue(report["finite_reference"]["public_state_immutable"])

    def test_float32_conversion_retains_source_dtype_and_uncertified_precision(self):
        high,low,bound,report,spec,expected=captured("linear32")
        self.assertEqual(set(report["capture_metadata"]["source_state_dtypes"].values()),{"float32"})
        self.assertFalse(report["source_precision_execution_compared"])

    def test_source_collision_and_missing_trust_rejected(self):
        _,_,_,_,spec,_=captured("linear")
        bad=copy.deepcopy(spec)
        text="def build(): return None"
        bad["files"]={"pytorch_ordered_capture.py":hashlib.sha256(text.encode()).hexdigest()}
        bad["entry"]="pytorch_ordered_capture.build"
        with self.assertRaisesRegex(ValueError,"conflicts"):
            validate_package(bad,{"pytorch_ordered_capture.py":text})

    def test_actual_trace_policy_cannot_be_relabelled_index(self):
        high,_,_,report,spec,expected=captured("linear")
        md=copy.deepcopy(report["capture_metadata"])
        md["ordered_alias_capture"]["source_capture_policy"]=md["registered_emission_policy"]
        md["ordered_alias_capture"]["binding"]=digest({k:v for k,v in md["ordered_alias_capture"].items() if k!="binding"})
        with self.assertRaisesRegex(ValueError,"Actual ordered trace policy"):
            check_capture_metadata(md,high,spec,resource_profile=PROFILE,expected_weight_binding=expected)

    def test_loaded_weight_content_is_independently_bound(self):
        high,_,_,report,spec,expected=captured("linear")
        fake=copy.deepcopy(report["capture_metadata"])
        state=next(iter(fake["weight_binding"]["normalized_float64_sha256"]))
        fake["weight_binding"]["normalized_float64_sha256"][state]="0"*64
        with self.assertRaisesRegex(ValueError,"Loaded weight content"):
            check_capture_metadata(fake,high,spec,resource_profile=PROFILE,expected_weight_binding=expected)

    def test_original_dtype_inventory_tamper_rejected(self):
        high,_,_,report,spec,expected=captured("linear32")
        fake=copy.deepcopy(report["capture_metadata"])
        key=next(iter(fake["source_state_dtypes"]))
        fake["source_state_dtypes"][key]="float64"
        with self.assertRaisesRegex(ValueError,"Original state dtype inventory"):
            check_capture_metadata(fake,high,spec,resource_profile=PROFILE,expected_weight_binding=expected)

    def test_public_view_contents_cannot_be_rebound_away_from_original_weight(self):
        high,_,_,report,spec,expected=captured("transpose_weight")
        fake=copy.deepcopy(report["capture_metadata"])
        alias=fake["ordered_alias_capture"]
        view=next(iter(alias["capture_public_constant_views"].values()))
        indices=np.asarray(view["indices"]).ravel()
        indices[0]=(indices[0]+1)%np.prod(view["source_shape"])
        view["indices"]=indices.reshape(view["result_shape"]).tolist()
        alias["binding"]=digest({k:v for k,v in alias.items() if k!="binding"})
        with self.assertRaisesRegex(ValueError,"Public emission view differs"):
            check_capture_metadata(fake,high,spec,resource_profile=PROFILE,expected_weight_binding=expected)

    def test_source_graph_model_and_context_hash_tamper_rejected(self):
        high,_,_,report,spec,expected=captured("linear")
        changed=copy.deepcopy(high)
        changed["id"]="tampered"
        with self.assertRaisesRegex(ValueError,"source/runtime"):
            check_capture_metadata(report["capture_metadata"],changed,spec,
                                   resource_profile=PROFILE,expected_weight_binding=expected)

    def test_matrix_plan_tamper_rejected_after_metadata_rebinding(self):
        high,_,_,report,spec,expected=captured("matrix")
        changed=copy.deepcopy(report["capture_metadata"])
        entry=changed["matrix_capture"]["entries"][0]
        entry["plan"]["operation"]="dot"
        with self.assertRaises(ValueError):
            check_capture_metadata(changed,high,spec,resource_profile=PROFILE,expected_weight_binding=expected)

    def test_index_plan_tamper_rejected_after_metadata_rebinding(self):
        high,_,_,report,spec,expected=captured("linear")
        changed=copy.deepcopy(report["capture_metadata"])
        entry=changed["index_capture"]["entries"][0]
        entry["source_shape"]=[9,9]
        with self.assertRaisesRegex(ValueError,"Index source shape"):
            check_capture_metadata(changed,high,spec,resource_profile=PROFILE,expected_weight_binding=expected)

    def test_reference_tolerance_and_compile_scope_cannot_be_upgraded(self):
        high,low,_,report,spec,expected=captured("linear")
        for field,value in (("encrypted_execution",True),("probes",15)):
            changed=copy.deepcopy(report);changed[field]=value
            with self.assertRaises(ValueError):
                check_reference(changed,high,low,spec,resource_profile=PROFILE,expected_weight_binding=expected)
        changed=copy.deepcopy(report)
        finite=changed["finite_reference"]
        finite["comparison_tolerance"]={"atol":1e-3,"rtol":1e-3}
        finite["binding"]=digest({k:v for k,v in finite.items() if k!="binding"})
        with self.assertRaisesRegex(ValueError,"Finite ordered reference"):
            check_reference(changed,high,low,spec,resource_profile=PROFILE,expected_weight_binding=expected)

    def test_numerical_failure_cannot_be_relabelled_passed(self):
        high,low,_,report,spec,expected=captured("linear")
        changed=copy.deepcopy(report)
        finite=changed["finite_reference"];finite["max_reference_tolerance_ratio"]=1.1
        finite["binding"]=digest({k:v for k,v in finite.items() if k!="binding"})
        with self.assertRaisesRegex(ValueError,"comparison failed"):
            check_reference(changed,high,low,spec,resource_profile=PROFILE,expected_weight_binding=expected)

    def test_input_public_state_and_overlap_rejections(self):
        for name in ("public_mutation","overlap","dynamic","linear32_strict"):
            package=Path("/fixtures")/name
            spec,_=prepare(package/"manifest.json")
            out=Path("/out")/("negative-"+name);out.mkdir()
            request=out/"request.json"
            request.write_text(json.dumps(dict(manifest=spec,resource_profile=PROFILE,
                 runtime=legacy.runtime_identity(),source_hashes=source_identity())))
            env={k:v for k,v in os.environ.items() if k in (
                 "LD_LIBRARY_PATH","POSEIDON_PLATFORM","POSEIDON_WORK_ROOT","HOME",
                 "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","LANG")}
            env.update(PYTHONPATH="/app:"+str(package),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
            code="from pytorch_ordered_intake import capture_request;capture_request(*__import__('sys').argv[1:])"
            p=subprocess.run([sys.executable,"-B","-c",code,str(request),str(package),str(out)],
                             env=env,text=True,capture_output=True,timeout=25)
            (out/"worker.log").write_text(p.stdout+p.stderr)
            self.assertNotEqual(p.returncode,0,name)
            self.assertFalse((out/"capture-reference.json").exists(),name)

    def test_closure_has_transitive_reference_and_platform_configuration(self):
        identity=source_identity()
        for name in ("operator_graph.py","benchmark_math.py","benchmark_torch.py",
                     "pytorch_ordered_reference.py","platform-profiles.json","tensor_expansion.py"):
            self.assertIn(name,identity)
        self.assertEqual(len(identity),len(set(identity)))
