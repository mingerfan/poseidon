"""Frozen source metadata stays identical after the existing JSON writer."""
import copy
import json
import unittest
from unittest.mock import patch
from pt_agent.experiments import qwen_prefill_compile as entry
from pt_agent.experiments import qwen_layer_native_probe as probe
from pt_agent.tests import test_qwen_prefill_compile as fixtures


class FrozenSourceSerializationChecks(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.QwenPrefillCompileChecks(methodName='runTest')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.case,self.source,self.outcome=self.fixture.source_fixture()
        self.source['recovery']=dict(pinned_error_counts={8:43},nested={9:('kept',1.25,False)})
        record=entry.read(self.case/'result.json');record['source']=self.source
        fixtures.save(self.case/'result.json',record)

    def frozen(self,function=entry.frozen_source):
        with (patch.object(entry.bounded,'session_outcome',return_value=self.outcome),
              patch.object(entry.bounded,'source_gate',return_value=self.source)):
            return function(self.case)

    def test_returned_original_source_is_json_stable_without_mutating_original(self):
        before=copy.deepcopy(self.source);record=self.frozen()
        self.assertEqual(record,json.loads(json.dumps(record,allow_nan=False)))
        self.assertEqual(record['original_source']['recovery']['pinned_error_counts'],{'8':43})
        self.assertEqual(record['original_source']['recovery']['nested'],{'9':['kept',1.25,False]})
        self.assertEqual(self.source,before)

    def test_real_probe_admission_survives_save_and_readback(self):
        record=self.frozen(probe.admission)
        fixtures.save(self.case/'saved-probe-source.json',record)
        self.assertEqual(entry.read(self.case/'saved-probe-source.json'),record)
        self.assertTrue(record['candidate_frozen']);self.assertFalse(record['full_model_compile_verified'])
        self.assertEqual(record['probe_layer_index'],0)

    def test_numeric_metadata_drift_remains_rejected(self):
        record=entry.read(self.case/'result.json')
        record['source']['recovery']['pinned_error_counts']['8']=44
        fixtures.save(self.case/'result.json',record)
        with self.assertRaisesRegex(ValueError,'metadata differs'):self.frozen()


if __name__=='__main__':unittest.main()
