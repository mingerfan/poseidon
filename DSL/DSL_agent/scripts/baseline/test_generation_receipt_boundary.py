"""Data-only provenance and file boundary regressions; no network or FHE."""
import copy, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from generation_protocol import Config, ProviderError
from generation_receipt import verify
from strict_json_files import strict_file, dump
from benchmark_graph import digest

class JsonFilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
    def test_duplicate_nonfinite_size_and_indirect_files_rejected(self):
        p=self.root/"data.json"
        for raw in [b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}']:
            p.write_bytes(raw)
            with self.assertRaises(ValueError):strict_file(p,100)
        p.write_bytes(b'{"x":1}')
        with self.assertRaises(ValueError):strict_file(p,2)
        link=self.root/"indirect.json";link.symlink_to(p)
        with self.assertRaises(ValueError):strict_file(link,100)
        with self.assertRaises(ValueError):strict_file(self.root,100)
    def test_writer_keeps_wire_format_and_rejects_nonfinite(self):
        p=self.root/"data.json";value={"z":[1,-2.5],"a":"value"}
        expected=json.dumps(value,indent=2,sort_keys=True,allow_nan=False).encode()+b"\n"
        dump(p,value)
        self.assertEqual(p.read_bytes(),expected)
        self.assertEqual(strict_file(p,1000),value)
        self.assertFalse(p.with_name(p.name+".partial").exists())
        with self.assertRaises(ValueError):dump(p,{"x":float("nan")})
        self.assertEqual(p.read_bytes(),expected)
    def test_benchmark_aliases_still_share_same_reader_and_writer(self):
        from benchmark_runner import strict_file as legacy_read, dump as legacy_write
        self.assertIs(legacy_read,strict_file)
        self.assertIs(legacy_write,dump)

class ReceiptBoundaryTests(unittest.TestCase):
    def test_legacy_receipt_exports_are_exact_shared_functions(self):
        from application_generation import verify as legacy
        from deepseek_provider import Config as legacy_config, ProviderError as legacy_error
        self.assertIs(legacy,verify)
        self.assertIs(legacy_config,Config)
        self.assertIs(legacy_error,ProviderError)
    def test_protocol_configuration_limits_and_exact_usage_rejection(self):
        from generation_protocol import usage_counts
        for change,code in [
            ({"model":"other"},"unsupported_model"),({"service_provider":"other"},"unsupported_provider"),
            ({"max_tokens":384001},"output_token_limit"),({"max_tokens":True},"output_token_limit"),
            ({"max_calls":5},"call_limit"),({"timeout_seconds":float("nan")},"timeout_limit"),
            ({"provider_retries":4},"retry_limit"),({"stream":1},"invalid_stream_setting")]:
            with self.subTest(change=change),self.assertRaisesRegex(ProviderError,code):
                Config(**change)
        counts=usage_counts({"usage":{"prompt_tokens":100,"completion_tokens":30,"total_tokens":130,
             "prompt_cache_hit_tokens":20,"prompt_cache_miss_tokens":80,
             "completion_tokens_details":{"reasoning_tokens":10}}},Config())
        self.assertEqual(counts["reasoning_tokens"],10)
        for bad in [
            {"prompt_tokens":100,"completion_tokens":30,"total_tokens":129},
            {"prompt_tokens":True,"completion_tokens":30,"total_tokens":31},
            {"prompt_tokens":100,"completion_tokens":9000,"total_tokens":9100},
        ]:
            with self.assertRaises(ProviderError):usage_counts({"usage":bad},Config())
    def test_data_only_imports_do_not_load_generator_or_developer_runner(self):
        script = """
import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname.split('.')[0] in {'deepseek_provider','deepseek_http_worker','application_generation','benchmark_runner'}:
            raise AssertionError('forbidden import: '+fullname)
sys.meta_path.insert(0,Guard())
import generation_receipt, generation_protocol, strict_json_files, candidate_bundle, model_import_cli
from component_backend import qualify
assert not any(n in sys.modules for n in ('deepseek_provider','application_generation','benchmark_runner'))
"""
        script="import sys;sys.path.insert(0,"+repr(str(Path(__file__).parent))+")\n"+script
        child=subprocess.run([sys.executable,"-B","-c",script],env=dict(os.environ),
                             capture_output=True,text=True,timeout=30)
        self.assertEqual(child.returncode,0,child.stderr)

if __name__=="__main__":unittest.main()
