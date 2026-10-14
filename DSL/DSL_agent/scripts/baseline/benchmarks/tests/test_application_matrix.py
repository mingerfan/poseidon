"""Offline application-matrix controls, synthetic plans only."""
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
TOOL=Path(__file__).resolve().parents[1]/"tools/run_application_matrix.py"
spec=importlib.util.spec_from_file_location("matrix_fixture",TOOL)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from benchmark_graph import digest
def fixture():
    value=dict(format=module.FORMAT,cases=[{"id":"fixture-only"}],source_hashes={},backend_identity={},
               application_identities={},tool_sha256="a"*64,provider=copy.deepcopy(module.PROVIDER),
               limits=copy.deepcopy(module.BOUNDS),max_generations=4,max_http_attempts=16)
    value["binding"]=digest(value)
    return value
def resign(value):
    value["binding"]=digest({k:v for k,v in value.items() if k!="binding"})
class MatrixControls(unittest.TestCase):
    def test_approved_plan_has_bounded_calls_and_serial_execution(self):
        p=fixture();self.assertEqual(module.check_envelope(p,p["binding"]),p)
        self.assertEqual(p["limits"]["api_concurrency"],1)
        self.assertEqual(p["limits"]["native_concurrency"],1)
    def test_binding_and_budget_mutations_are_rejected(self):
        for field in ("binding","budget","provider","calls"):
            p=fixture()
            if field=="binding":p["binding"]="f"*64
            if field=="budget":p["limits"]["case_bytes"]*=2;resign(p)
            if field=="provider":p["provider"]["max_repairs"]=4;resign(p)
            if field=="calls":p["max_http_attempts"]=17;resign(p)
            with self.assertRaises(ValueError):module.check_envelope(p,p["binding"])
    def test_empty_or_oversize_shard_is_rejected(self):
        for count in (0,49):
            p=fixture();p["cases"]=[{}]*count;p["max_generations"]=count*4;p["max_http_attempts"]=count*16;resign(p)
            with self.assertRaises(ValueError):module.check_envelope(p,p["binding"])
    def test_prepare_uses_no_provider_or_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"cases.json";output=root/"plan.json";source.write_text("[]")
            with patch.object(sys,"argv",["tool","--prepare",str(source),"--output",str(output),"--inside"]), \
                 patch.dict("os.environ",IN_NIX_SHELL="pure"),patch.object(module,"freeze",return_value=fixture()), \
                 patch("agent_credentials.load_api_key",side_effect=AssertionError("credential forbidden")), \
                 patch("deepseek_provider.HTTPSTransport",side_effect=AssertionError("transport forbidden")):
                self.assertEqual(module.main(),0)
            self.assertEqual(json.loads(output.read_text())["max_http_attempts"],16)
    def test_prepare_preserves_existing_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"cases.json";output=root/"plan.json";source.write_text("[]");output.write_text("original")
            with patch.object(sys,"argv",["tool","--prepare",str(source),"--output",str(output),"--inside"]), \
                 patch.dict("os.environ",IN_NIX_SHELL="pure"):
                with self.assertRaises(ValueError):module.main()
            self.assertEqual(output.read_text(),"original")
