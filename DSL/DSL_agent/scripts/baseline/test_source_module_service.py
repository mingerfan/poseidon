"""Explicit v2 application dispatch and public source summary controls."""
import copy,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
from benchmark_graph import digest
from source_preparation_service import SourcePreparation,REQUEST,MODULE_REQUEST,MODULE_RESULT,public_summary
class ModuleServiceTests(unittest.TestCase):
    def request(self):
        return dict(format=MODULE_REQUEST,action="inspect",
          manifest=dict(path="/home/lhohy/Code/Poseidon/manifest.json",sha256="a"*64),
          source_signature=dict(format="poseidon-source-calibration-signature-v1",outputs=[dict(name="out",shape=[2])]),
          samples=[dict(x=[.1,.2])],input_dtypes={"x":"float64"},
          output_structure={"kind":"tensor","name":"out"},resource_profile="general-tensors-v1")
    def receipt(self):
        return dict(status="described",binding="b"*64,fx_graph=dict(format="poseidon-source-calibration-fx-graph-v2",
            graph_sha256="c"*64,nodes=[dict(name="activation",op="call_module",target="activation",
            args="private_original_args",kwargs="private_original_kwargs",source=dict(module_paths=["activation"],
            module=dict(type="torch.nn.GELU",attributes=dict(approximate="none"))))]))
    def test_explicit_v2_inspect_requests_original_module_metadata(self):
        with patch("source_preparation_service.inspect_source",return_value=self.receipt())as inspect:
            result=SourcePreparation().run(self.request(),"/home/lhohy/poseidon-work/platforms/aarch64-linux/results/unit",trust_local_code=True)
        self.assertEqual(result["status"],"described");self.assertEqual(result["format"],MODULE_RESULT)
        self.assertTrue(inspect.call_args.kwargs["source_locations"])
    def test_legacy_v1_dispatch_keeps_old_graph_description(self):
        value=self.request();value["format"]=REQUEST
        with patch("source_preparation_service.inspect_source",return_value=dict(status="described",binding="b"*64))as inspect:
            result=SourcePreparation().run(value,"/home/lhohy/poseidon-work/platforms/aarch64-linux/results/unit",trust_local_code=True)
        self.assertEqual(result["status"],"described")
        self.assertFalse(inspect.call_args.kwargs["source_locations"])
    def test_public_v2_inventory_contains_module_metadata_without_arguments(self):
        result=dict(format=MODULE_RESULT,status="described",receipt=self.receipt())
        summary=public_summary(result);node=summary["node_inventory"][0]
        self.assertEqual(node["source"]["module"]["type"],"torch.nn.GELU")
        self.assertNotIn("private_original",str(summary))
        self.assertNotIn("samples",summary)
if __name__=="__main__":unittest.main(verbosity=2)
