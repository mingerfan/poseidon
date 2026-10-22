"""Ordered capture must survive task storage without changing old identities."""
import copy,json,tempfile,unittest
from pathlib import Path
from benchmark_graph import canonical,digest
from application_component import Application

class ProvenanceTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_pytorch_mha_component import OrderedMhaComponent
        OrderedMhaComponent.setUpClass()
        source=OrderedMhaComponent()
        high,_,binding,_=source.prepare(source.module("masked"),["value","weights"])
        cls.high=high
        cls.capture=binding["pytorch_capture"]
        cls.payload={"request":{"model":high},"model_provenance":{
            "original":high,"reference_report":{"capture_metadata":cls.capture}}}
        cls.payload["binding"]=digest(cls.payload)

    @staticmethod
    def reconstruct(payload):
        from pytorch_alias_capture_bridge import normalize_emissions
        alias=payload["model_provenance"]["reference_report"]["capture_metadata"]["ordered_alias_capture"]
        return normalize_emissions(alias["capture_emissions"],alias["capture_source_origins"],
            resource_profile="general-tensors-v1",
            constant_dtypes=alias["capture_emission_constant_dtypes"],
            state_constant_sources=alias["capture_emission_state_constant_sources"],
            public_constant_views=alias["capture_public_constant_views"])

    def test_capture_graph_round_trip_uses_original_constant_coordinates(self):
        with tempfile.TemporaryDirectory() as t:
            app=Application(Path(t)/"tasks",None);path=app.store/"task.json"
            before=self.reconstruct(self.payload)
            app._write(path,copy.deepcopy(self.payload));loaded=app._load(path)
            after=self.reconstruct(loaded)
            self.assertEqual(canonical(before[0]),canonical(self.high))
            self.assertEqual(canonical(after),canonical(before))
            self.assertEqual(digest(loaded),digest(self.payload))
            self.assertEqual(loaded["binding"],self.payload["binding"])

    def test_old_sorting_changes_capture_normalization_proof(self):
        sorted_payload=json.loads(canonical(self.payload))
        self.assertNotEqual(canonical(self.reconstruct(sorted_payload)),canonical(self.reconstruct(self.payload)))

    def test_graph_only_task_keeps_original_canonical_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            app=Application(Path(t)/"tasks",None);path=app.store/"task.json"
            value={"z":2,"request":{"b":1,"a":0}}
            app._write(path,value)
            self.assertEqual(path.read_bytes(),canonical(value))

    def test_status_and_ledger_serialization_remains_canonical(self):
        with tempfile.TemporaryDirectory() as t:
            app=Application(Path(t)/"tasks",None);path=app.store/"state.json"
            app._write(path,self.payload)
            self.assertEqual(path.read_bytes(),canonical(self.payload))

    def test_nonfinite_provenance_cannot_be_persisted(self):
        with tempfile.TemporaryDirectory() as t:
            app=Application(Path(t)/"tasks",None);path=app.store/"task.json"
            with self.assertRaises(ValueError):
                app._write(path,{"model_provenance":{"constant":float("nan")}})
            self.assertFalse(path.exists())

if __name__=="__main__":unittest.main()
