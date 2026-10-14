"""Parameter boundary controls; no compiler, FHE executor or Provider calls."""
import ast,copy,hashlib,json,resource,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from application_planning import backend_parameters as isolated
from application_planning import resources as legacy
from application_identity import source_closure
from compiler_configuration import request_configuration
from model_weight_package import regular_bytes
from poseidon_runtime_paths import SPEC
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"scripts/baseline"
TOOLS=BASE/"benchmarks/tools/resource_parameter_boundary_v1"
PROPOSAL=ROOT/"docs/baseline/agent-mha-feedback-live-proposal-r1317.json"
STAGES=("generation","compilation","execution","validation")
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class ParameterBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proposal=read(PROPOSAL)
        cls.requests=[read(Path(c["request_file"])) for c in cls.proposal["cases"]]
        cls.graphs={}
    def test_real_parameters_equal_legacy_for_both_reviewed_models(self):
        for request in self.requests:
            with self.subTest(request=request["request_id"]):
                self.assertEqual(isolated.parameters(request),legacy.parameters(request))
                value=isolated.parameters(request)
                self.assertEqual(value["data_modulus_count"],13)
                self.assertEqual(value["poly_degree"],32768)
                self.assertEqual(value["input_level"],12)
                self.assertEqual(value["input_components"],2)
                self.assertEqual(value["coefficient_bytes"],8)
                self.assertEqual(value["operator_spec_sha256"],sha(SPEC))
                for name,expected in value["source_basis_sha256"].items():
                    self.assertEqual(sha(ROOT/name),expected)
    def test_native_limit_is_exact_and_does_not_modify_host_rlimits(self):
        before=resource.getrlimit(resource.RLIMIT_AS)
        self.assertEqual(isolated.native_address_limit(),legacy.native_address_limit())
        self.assertEqual(isolated.native_address_limit(),4*1024**3)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS),before)
    def test_literal_limit_arithmetic_rejects_nonliteral_or_unbounded_values(self):
        for expression in ("True","-1","limit","2+3","2**17","2**16*2**16*2**16*2",
                           "__import__('os').system('unused')","2**49","2**'3'"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    isolated._integer(ast.parse(expression,mode="eval").body)
        for expression,expected in (("0",0),("4*1024**3",4*1024**3),("2**16",2**16)):
            self.assertEqual(isolated._integer(ast.parse(expression,mode="eval").body),expected)
    def test_address_policy_requires_one_equal_positive_soft_and_hard_limit(self):
        headers="def limits():\n"
        call="    resource.setrlimit(resource.RLIMIT_AS, {values})\n"
        for body in (
            "    pass\n",
            call.format(values="(1,2)"),
            call.format(values="(0,0)"),
            call.format(values="[4,4]"),
            call.format(values="(True,True)"),
            call.format(values="(2**17,2**17)"),
            call.format(values="(4,4)")+call.format(values="(4,4)")):
            with self.subTest(body=body):
                with patch("inspect.getsource",return_value=headers+body):
                    with self.assertRaises(ValueError):isolated.native_address_limit()
    def test_spec_content_mismatch_rejected_before_parameter_use(self):
        original=regular_bytes
        def bytes_for(path,*args):
            return b"{}" if Path(path)==SPEC else original(path,*args)
        for implementation in (isolated,legacy):
            with self.subTest(module=implementation.__name__):
                with patch("model_weight_package.regular_bytes",side_effect=bytes_for):
                    with self.assertRaisesRegex(ValueError,"resource_spec_identity"):
                        implementation.parameters(self.requests[0])
    def test_parameter_shape_contract_is_not_relaxed(self):
        raw=regular_bytes(SPEC,65536);spec=json.loads(raw.decode())
        variants=[]
        bad=copy.deepcopy(spec);bad["context"]["poly_degree"]=16384;variants.append(bad)
        bad=copy.deepcopy(spec);bad["context"]["rns_moduli_log2"]=bad["context"]["rns_moduli_log2"][:-1];variants.append(bad)
        bad=copy.deepcopy(spec);bad["context"]["context_id"]="other";variants.append(bad)
        bad=copy.deepcopy(spec);bad["levels"]["lower_bound"]=1;variants.append(bad)
        bad=copy.deepcopy(spec);bad["levels"]["upper_bound"]=13;variants.append(bad)
        bad=copy.deepcopy(spec);bad["spec_id"]="other";variants.append(bad)
        original=regular_bytes
        for bad in variants:
            raw=json.dumps(bad,sort_keys=True,separators=(",",":")).encode()
            config=copy.deepcopy(request_configuration(self.requests[0]))
            config["operator_spec_sha256"]=hashlib.sha256(raw).hexdigest()
            def bytes_for(path,*args):
                return raw if Path(path)==SPEC else original(path,*args)
            for implementation in (isolated,legacy):
                with self.subTest(module=implementation.__name__,variant=variants.index(bad)):
                    with patch("compiler_configuration.request_configuration",return_value=config), \
                         patch("model_weight_package.regular_bytes",side_effect=bytes_for):
                        with self.assertRaisesRegex(ValueError,"resource_parameter_contract"):
                            implementation.parameters(self.requests[0])
    def test_other_backend_is_rejected(self):
        config=copy.deepcopy(request_configuration(self.requests[0]));config["backend"]="seal_cpu"
        for implementation in (isolated,legacy):
            with patch("compiler_configuration.request_configuration",return_value=config):
                with self.assertRaisesRegex(ValueError,"resource_backend"):
                    implementation.parameters(self.requests[0])
    def test_new_module_has_complete_provider_free_source_closure(self):
        closure=source_closure(BASE,("application_planning/backend_parameters.py",))
        self.assertIn("process_resource_limits.py",closure["sources"])
        for forbidden in ("sandbox_policy.py","application_planning/layout.py",
                          "application_component.py","application_generation.py",
                          "deepseek_provider.py","agent_credentials.py",
                          "run_agent_batch.py","run_candidate.py"):
            self.assertNotIn(forbidden,closure["sources"])
        self.graphs["parameter_module_sources"]=len(closure["sources"])
    def test_activation_patch_is_clean_and_current_source_is_frozen(self):
        result=subprocess.run(["git","apply","--check",str(TOOLS/"activate.patch")],
                              cwd=ROOT,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        expected=read(TOOLS/"source-before.json")
        for name,value in expected.items():self.assertEqual(sha(ROOT/name),value)
        for stage in STAGES:
            closure=source_closure(BASE,tuple(self.proposal["source_stage_identity"][stage]["roots"]))
            self.assertEqual(closure["sources"],self.proposal["source_stage_identity"][stage]["sources"])
    def test_patch_removes_backedges_without_changing_old_interface(self):
        # This is a source-only copy of Agent files, not a development checkout
        # and not an execution environment. Only complete source closures run.
        inventory=self.proposal["source_stage_identity"]["generation"]["sources"]
        with tempfile.TemporaryDirectory(prefix="closure-",dir=TOOLS) as folder:
            copy_root=Path(folder);copy_base=copy_root/"scripts/baseline"
            for name in inventory:
                dest=copy_base/name;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes((BASE/name).read_bytes())
            name="application_planning/backend_parameters.py"
            (copy_base/name).write_bytes((BASE/name).read_bytes())
            for stage in STAGES:
                roots=tuple(self.proposal["source_stage_identity"][stage]["roots"])
                before=source_closure(copy_base,roots)
                self.assertEqual(before["sources"],self.proposal["source_stage_identity"][stage]["sources"])
            result=subprocess.run(["git","apply",str(TOOLS/"activate.patch")],
                                  cwd=copy_root,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            source=(copy_base/"application_planning/resources.py").read_text()
            imported={alias.name for node in ast.walk(ast.parse(source))
                      if isinstance(node,ast.ImportFrom)
                      and node.module=="application_planning.backend_parameters"
                      for alias in node.names}
            self.assertEqual(imported,{"_integer","native_address_limit","parameters"})
            for stage in STAGES:
                roots=tuple(self.proposal["source_stage_identity"][stage]["roots"])
                after=source_closure(copy_base,roots)
                self.assertIn(name,after["sources"])
                if stage!="generation":
                    for forbidden in ("application_component.py","application_generation.py",
                                      "deepseek_provider.py","agent_credentials.py","run_agent_batch.py",
                                      "run_candidate.py","sandbox_policy.py","application_planning/layout.py"):
                        self.assertNotIn(forbidden,after["sources"])
                self.graphs[stage]=dict(before=len(inventory),after=len(after["sources"]))
