"""Input-domain policy tests. Mocked execution tests do not perform FHE."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from benchmark_graph import canonical, digest
from model_decomposition import FORMAT, decompose
from execution_input_domain import (
    ApproximationDomainRejected, check_approximation_domains, declared_profiles)
from program_execution import normalize_samples, pack_samples, package_execution_metadata
from component_contract import prepare_task
from benchmark_suite import Builder


def graph(op, attrs, shape=(2,), constants=None, refs=None):
    return dict(format=FORMAT, id="domain_fixture",
        inputs=[dict(name="x", shape=list(shape))], constants=constants or {},
        nodes=[dict(id="n", op=op, inputs=refs or ["x"], attrs=attrs, outputs=["y"])],
        outputs=[dict(name="output", value="y")])


def approximation(op, domain=None):
    return dict(method={"exp":"taylor","reciprocal":"goldschmidt","rsqrt":"newton"}[op],
                domain=domain or ([-1.,1.] if op=="exp" else [1.,5.]),
                **{("degree" if op=="exp" else "iterations"):1})


def polynomial(domain):
    return dict(method="polynomial", basis="power", domain=domain, coefficients=[0.,0.,1.])


class DomainTests(unittest.TestCase):
    def checked(self, model, samples):
        decompose(model)
        return check_approximation_domains(model, samples)

    def rejects(self, model, value, index=0, sample_index=0):
        samples=value if type(value) is list else [dict(x=value)]
        decompose(model)
        with self.assertRaises(ApproximationDomainRejected) as captured:
            check_approximation_domains(model,samples)
        error=captured.exception
        self.assertEqual(error.public_failure["location"]["index"],index)
        self.assertEqual(error.public_failure["context"]["sample_index"],sample_index)
        return error

    def test_exact_graph_is_explicitly_not_declared(self):
        model=graph("square",{})
        with patch("operator_reference.evaluate",side_effect=AssertionError("unneeded reference")):
            checked=self.checked(model,[dict(x=np.zeros(2))])
        self.assertEqual(checked["status"],"not_declared")
        self.assertEqual(checked["profile_count"],0)

    def test_three_shapes_and_closed_boundaries(self):
        for shape in ((2,),(1,2),(1,1,2)):
            for op in ("exp","reciprocal","rsqrt"):
                with self.subTest(shape=shape,op=op):
                    profile=approximation(op)
                    model=graph(op,dict(approximation=profile),shape)
                    sample=np.asarray(profile["domain"],dtype=np.float64).reshape(shape)
                    checked=self.checked(model,[dict(x=sample)])
                    self.assertEqual(checked["status"],"passed")
                    self.assertEqual(checked["node_count"],1)

    def test_one_ulp_outside_boundaries_is_rejected(self):
        for op in ("exp","reciprocal","rsqrt"):
            p=approximation(op);model=graph(op,dict(approximation=p))
            for value in (np.nextafter(p["domain"][0],-np.inf),np.nextafter(p["domain"][1],np.inf)):
                with self.subTest(op=op,value=value):
                    self.rejects(model,np.full(2,value))

    def test_rms_norm_checks_derived_variance(self):
        model=graph("rms_norm",dict(eps=1.,approximation=approximation("rsqrt")),
                    constants=dict(gain=[1.,1.]),refs=["x","gain"])
        self.assertEqual(self.checked(model,[dict(x=np.zeros(2))])["status"],"passed")
        self.rejects(model,np.array([10.,-10.]))

    def test_layer_norm_three_contexts(self):
        for shape in ((2,),(1,2),(2,2)):
            with self.subTest(shape=shape):
                model=graph("layer_norm",dict(eps=1.,normalized_shape=[2],
                    approximation=approximation("rsqrt")),shape,
                    constants=dict(gain=[1.,1.],bias=[0.,0.]),refs=["x","gain","bias"])
                self.checked(model,[dict(x=np.zeros(shape))])
                invalid=np.tile(np.array([0.,10.]),int(np.prod(shape))//2).reshape(shape)
                self.rejects(model,invalid)

    def test_group_norm_three_contexts(self):
        for shape in ((1,2,2),(1,2,3),(2,2,2)):
            with self.subTest(shape=shape):
                model=graph("group_norm",dict(eps=1.,groups=1,approximation=approximation("rsqrt")),
                    shape,constants=dict(gain=[1.,1.],bias=[0.,0.]),refs=["x","gain","bias"])
                self.checked(model,[dict(x=np.zeros(shape))])
                bad=np.arange(np.prod(shape),dtype=np.float64).reshape(shape)*10
                self.rejects(model,bad)

    def test_instance_norm_checks_derived_variance(self):
        model=graph("instance_norm",dict(eps=1.,approximation=approximation("rsqrt")),
                    (1,2,2),constants=dict(gain=[1.,1.],bias=[0.,0.]),refs=["x","gain","bias"])
        self.checked(model,[dict(x=np.zeros((1,2,2)))])
        self.rejects(model,np.array([[[0.,10.],[0.,10.]]]))

    def test_sqrt_checks_its_positive_domain(self):
        model=graph("sqrt",dict(approximation=approximation("rsqrt")))
        self.checked(model,[dict(x=np.array([1.,5.]))])
        self.rejects(model,np.array([0.,1.]))

    def test_softmax_checks_denominator_after_exp(self):
        attrs=dict(axis=0,exp=approximation("exp",[-2.,2.]),
                   reciprocal=approximation("reciprocal",[1.,1.5]))
        model=graph("softmax",attrs)
        # Each input is within the exp interval; the exp sum is outside reciprocal's.
        error=self.rejects(model,np.zeros(2))
        self.assertEqual(set(error.public_failure["context"]["declared_domains"]),{"exp","reciprocal"})

    def test_softmax_valid_denominator(self):
        model=graph("softmax",dict(axis=0,exp=approximation("exp",[-1.,1.]),
                                  reciprocal=approximation("reciprocal",[1.,4.])))
        checked=self.checked(model,[dict(x=np.array([-.1,.1]))])
        self.assertEqual(checked["profile_count"],2)

    def test_logsoftmax_checks_log_domain_of_exp_sum(self):
        model=graph("logsoftmax",dict(axis=0,exp=approximation("exp",[-1.,1.]),
                    log=polynomial([1.,1.5])))
        self.rejects(model,np.zeros(2))

    def test_polynomial_function_domains(self):
        for op in ("relu","gelu","silu","sigmoid","tanh","leaky_relu"):
            attrs=dict(approximation=polynomial([-1.,1.]))
            if op=="leaky_relu":attrs["negative_slope"]=.1
            with self.subTest(op=op):
                model=graph(op,attrs)
                self.checked(model,[dict(x=np.array([-.5,.5]))])
                self.rejects(model,np.array([0.,1.1]))

    def test_prelu_domain_uses_input_before_gain(self):
        model=graph("prelu",dict(approximation=polynomial([-1.,1.])),
                    constants=dict(slope=[.1]),refs=["x","slope"])
        self.rejects(model,np.array([0.,2.]))

    def test_log_requires_its_declared_positive_interval(self):
        model=graph("log",dict(approximation=polynomial([.1,2.])))
        self.rejects(model,np.array([0.,1.]))

    def test_pair_extrema_use_difference_not_each_operand(self):
        for op in ("maximum","minimum"):
            model=graph(op,dict(approximation=polynomial([-1.,1.])),
                        constants=dict(rhs=[-.6,-.6]),refs=["x","rhs"])
            with self.subTest(op=op):
                self.rejects(model,np.array([.6,.6]))

    def test_reduction_extrema_check_later_tree_intermediates(self):
        for op in ("max","min"):
            model=graph(op,dict(axes=[0],keepdims=False,approximation=polynomial([-1.,1.])),(4,))
            with self.subTest(op=op):
                self.rejects(model,np.array([0.,0.,1.1,1.1]))

    def test_maxpool_checks_comparison_domain(self):
        model=graph("max_pool1d",dict(kernel=[2],stride=[1],padding=[0],dilation=[1],
                    approximation=polynomial([-1.,1.])),(1,1,2))
        self.rejects(model,np.array([[[0.,2.]]]))

    def test_first_violating_node_is_located_after_exact_prefix(self):
        model=graph("exp",dict(approximation=approximation("exp")))
        model["constants"]["offset"]=[.8,.8]
        model["nodes"]=[
            dict(id="shift",op="add",inputs=["x","offset"],attrs={},outputs=["v"]),
            dict(id="first",op="exp",inputs=["v"],attrs=dict(approximation=approximation("exp")),outputs=["w"]),
            dict(id="second",op="exp",inputs=["w"],attrs=dict(approximation=approximation("exp")),outputs=["y"])]
        error=self.rejects(model,np.full(2,.4),index=1)
        self.assertEqual(error.public_failure["location"]["id"],"first")

    def test_every_sample_is_checked_with_stable_index(self):
        model=graph("exp",dict(approximation=approximation("exp")))
        samples=[dict(x=np.zeros(2)),dict(x=np.full(2,2.))]
        self.rejects(model,samples,sample_index=1)

    def test_sixteen_samples_and_original_snapshots_remain_unchanged(self):
        model=graph("exp",dict(approximation=approximation("exp")))
        before=copy.deepcopy(model);sample=np.zeros(2);samples=[dict(x=sample)]*16
        self.checked(model,samples)
        self.assertEqual(model,before);np.testing.assert_array_equal(sample,np.zeros(2))

    def test_sample_count_boundary(self):
        model=graph("exp",dict(approximation=approximation("exp")))
        for samples in ([],[dict(x=np.zeros(2))]*17):
            with self.assertRaises(ValueError):check_approximation_domains(model,samples)

    def test_profile_validation_is_strict(self):
        node=graph("exp",dict(approximation=approximation("exp")))["nodes"][0]
        for domain in ((-1.,1.),[1.,1.],[False,1.],[-np.inf,1.],[-1.,np.nan],[-1025.,1.]):
            bad=copy.deepcopy(node);bad["attrs"]["approximation"]["domain"]=domain
            with self.assertRaises(ValueError):declared_profiles(bad)

    def test_public_diagnostic_is_detached_and_redacted(self):
        model=graph("silu",dict(approximation=polynomial([-1.,1.])))
        error=self.rejects(model,np.array([0.,9.123456789]))
        diagnostic=error.public_failure
        diagnostic["context"]["declared_domains"]["approximation"][0]=-999.
        self.assertEqual(error.public_failure["context"]["declared_domains"]["approximation"],[-1.,1.])
        raw=json.dumps(error.public_failure)
        for forbidden in ("9.123456789","coefficients","reference","actual",'"inputs":',"/home/"):
            self.assertNotIn(forbidden,raw)

    def test_unknown_reference_error_is_not_disclosed_as_domain_failure(self):
        model=graph("exp",dict(approximation=approximation("exp")))
        with patch("operator_reference.evaluate",side_effect=ValueError("PRIVATE /env 9.123456789")):
            with self.assertRaises(ValueError) as error:self.checked(model,[dict(x=np.zeros(2))])
        self.assertNotIsInstance(error.exception,ApproximationDomainRejected)
        self.assertNotIn("PRIVATE",str(error.exception))
        self.assertNotIn("/env",str(error.exception))

    def test_approximate_intermediate_semantics_are_used(self):
        model=graph("exp",dict(approximation=approximation("exp")))
        model["nodes"].append(dict(id="second",op="exp",inputs=["y"],
            attrs=dict(approximation=approximation("exp",[.9,1.1])),outputs=["z"]))
        model["outputs"][0]["value"]="z"
        self.checked(model,[dict(x=np.zeros(2))])

    def test_post_domain_outputs_are_not_evaluated_or_certified(self):
        model=graph("exp",dict(approximation=approximation("exp")))
        model["nodes"].append(dict(id="late",op="square",inputs=["y"],attrs={},outputs=["z"]))
        model["outputs"][0]["value"]="z"
        with patch("execution_input_domain._evaluate_prefix",wraps=__import__("execution_input_domain")._evaluate_prefix) as call:
            result=self.checked(model,[dict(x=np.zeros(2))])
        self.assertEqual(call.call_args.args[2],1)
        self.assertNotIn("numerically_validated",result)
        self.assertNotIn("outputs",result)


class IntegrationTests(unittest.TestCase):
    def request(self):
        b=Builder([(2,)]);return prepare_task(b.finish(b.node("square",["input0"])))

    def test_normalized_arrays_are_copied_before_packing(self):
        req=self.request();original=np.array([1.,2.]);logical=normalize_samples(req,[dict(input0=original)])
        original[:]=9
        np.testing.assert_array_equal(logical[0]["input0"],[1.,2.])
        np.testing.assert_array_equal(pack_samples(req,[dict(input0=[1.,2.])])[0,0,:2],[1.,2.])

    def test_bound_original_model_is_reused_from_provenance(self):
        original=graph("exp",dict(approximation=approximation("exp")))
        lowered,binding=decompose(original)
        value=dict(original=original,decomposition=binding,reference_report=dict(
            original_sha256=digest(original),model_sha256=digest(lowered),
            reference_status="passed",approximation_accuracy_certified=False))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"model-provenance.json";raw=canonical(value);path.write_bytes(raw)
            manifest=dict(model_provenance="included",files={path.name:hashlib.sha256(raw).hexdigest()})
            checked=package_execution_metadata(folder,manifest,lowered)
            self.assertEqual(checked["original_model"],original)
            self.assertIsNone(checked["output_structure"])
            path.write_text("{}")
            with self.assertRaises(ValueError):package_execution_metadata(folder,manifest,lowered)

    def test_bound_graph_mismatch_is_rejected(self):
        original=graph("exp",dict(approximation=approximation("exp")))
        lowered,binding=decompose(original)
        changed=copy.deepcopy(original);changed["nodes"][0]["attrs"]["approximation"]["degree"]=2
        value=dict(original=changed,decomposition=binding,reference_report=dict(
            original_sha256=digest(original),model_sha256=digest(lowered),
            reference_status="passed",approximation_accuracy_certified=False))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"model-provenance.json";raw=canonical(value);path.write_bytes(raw)
            manifest=dict(model_provenance="included",files={path.name:hashlib.sha256(raw).hexdigest()})
            with self.assertRaises(ValueError):package_execution_metadata(folder,manifest,lowered)

    def test_no_provenance_has_no_invented_domain(self):
        req=self.request();result=package_execution_metadata(".",dict(model_provenance="absent"),req["model"])
        self.assertEqual(result["original_model"],req["model"])
        self.assertEqual(check_approximation_domains(result["original_model"],[dict(input0=np.zeros(2))])["status"],"not_declared")

    def test_invalid_domain_stops_before_identity_artifacts_or_runtime(self):
        import program_execution
        original=graph("exp",dict(approximation=approximation("exp")))
        lowered,_=decompose(original);request=prepare_task(lowered)
        with patch("candidate_bundle.check_execution_dependencies",return_value=({},request)), \
             patch("program_execution.package_execution_metadata",return_value=dict(original_model=original,output_structure=None)), \
             patch("poseidon_backend_identity.identity") as identity, \
             patch("tempfile.mkdtemp") as directory, \
             patch("poseidon_validation.execute") as runtime, \
             patch("native_execution_slots.native_slot") as slot:
            with self.assertRaises(ApproximationDomainRejected):
                program_execution.execute_program(".",[dict(x=[2.,0.])])
            for function in (identity,directory,runtime,slot):function.assert_not_called()

    def test_application_returns_typed_failure_without_repair_or_values(self):
        from application_component import Application
        original=graph("exp",dict(approximation=approximation("exp")))
        error=ApproximationDomainRejected(0,original["nodes"][0],{"approximation":[-1.,1.]},0)
        class Backend:
            def execute(self,*args):raise error
        with tempfile.TemporaryDirectory() as folder:
            app=Application(folder,Backend())
            with patch.object(app,"result",return_value=dict(status="succeeded",package=dict(task_id="a"*32))):
                result=app.execute("a"*32,[dict(x=[9.123456789,0.])])
        self.assertEqual(result["failure"]["code"],"approximation_domain_rejected")
        self.assertEqual(result["failure"]["layer"],"input")
        self.assertEqual(result["status"],"validation_failed")
        self.assertFalse(result["encrypted_execution"])
        self.assertEqual(result["agent_calls"],result["compiler_calls"])
        self.assertNotIn("9.123456789",json.dumps(result))


    def mocked_valid_execution(self, mutate_samples=False):
        """Exercise the real report builder with an injected, non-FHE executor."""
        import contextlib,program_execution
        original=graph("exp",dict(approximation=approximation("exp")))
        lowered,_=decompose(original);request=prepare_task(lowered)
        manifest=dict(binding="a"*64,validation=dict(level="numerical",
                      encrypted_execution=True,numerically_validated=True))
        logical=[dict(x=[0.,0.])]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);package=root/"program";package.mkdir()
            (package/"candidate.py").write_text('@hc.func("c,c")\ndef golden(x,zero_ct):\n    return [x]\n')
            evidence=root/"evidence";evidence.mkdir()
            def artifacts(folder,manifest,output):
                (output/"validation.json").write_text(json.dumps(dict(gate={})))
            def execute(payload,output,request,run,log):
                np.save(output/"decrypted.npy",np.ones((1,2),dtype=np.float64),allow_pickle=False)
                if mutate_samples:logical.append(dict(x=[9.,9.]))
                return dict(backend="poseidon_cpu",encrypted_execution=True,executor_secret_key=False,bootstrap=False)
            with patch("candidate_bundle.check_execution_dependencies",return_value=(manifest,request)), \
                 patch("program_execution.package_execution_metadata",return_value=dict(original_model=original,output_structure=None)), \
                 patch("poseidon_backend_identity.identity",return_value={}), \
                 patch("tempfile.mkdtemp",return_value=str(evidence)), \
                 patch("poseidon_package.copy_checked_artifacts",side_effect=artifacts), \
                 patch("poseidon_runtime_compile.audit_entrypoint"), \
                 patch("poseidon_validation.audit_gate"), \
                 patch("application_planning.compiled_resources.execution_storage_precheck", \
                       return_value=dict(status="assessed", execution_readiness="unknown", encrypted_execution=False)), \
                 patch("poseidon_validation.execute",side_effect=execute), \
                 patch("native_execution_slots.native_slot",return_value=contextlib.nullcontext()):
                return program_execution.execute_program(package,logical)

    def test_valid_mocked_execution_exposes_domain_check_without_numeric_claim(self):
        result=self.mocked_valid_execution()
        self.assertEqual(result["input_domain_check"]["status"],"passed")
        self.assertEqual(result["input_domain_check"]["samples"],1)
        self.assertFalse(result["numerically_validated_on_supplied_inputs"])
        self.assertFalse(result["application_accuracy_certified"])

    def test_valid_mocked_execution_reports_the_normalized_snapshot_count(self):
        result=self.mocked_valid_execution(mutate_samples=True)
        self.assertEqual(result["samples"],1)
        self.assertEqual(len(result["outputs"]),1)



if __name__=="__main__":unittest.main()
