from collections import Counter
import copy
import json
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.experiments import a4_repeated_calibration as own
from pt_agent.experiments import a4_calibration_run as runner
from pt_agent.experiments.r31_lattigo_profile39 import _canonical_base, build_profile, make_catalog_source
from pt_agent.experiments.r31_lattigo_profile_subset import EXPECTED_CELLS, deterministic_fixture
from pt_agent.experiments.r31_lattigo_profile_reference import compare_references, direct_profile_sum, replay_profile
from pt_agent.tests._artifacts import require


def fixture():
    base=dict(cell_count=36,cells=_canonical_base())
    missing=dict(cell_count=3,cells=[dict(c,representative_attrs=dict(steps=1023)
        if c['op']=='rotate' else c['attrs'])for c in EXPECTED_CELLS])
    plan,manifest,metadata,_=build_profile(base,missing,c2s_bits=52)
    return own.repeat_profile(plan,manifest,metadata)


def observations(metadata):
    return [dict(ordinal=t['target_ordinal'],cell_index=t['cell_index'],iteration=t['iteration'],
        operation_ns=2000+t['iteration'],compute_ns=1000+t['iteration'],completed=True)
        for t in metadata['target_operations']]


class RepeatedCalibrationTests(unittest.TestCase):
    def test_monitor_start_failure_preserves_original_error_and_durable_closure(self):
        class UnstartedMonitor:
            def __init__(self,**kwargs):pass
            def start(self):raise RuntimeError('fixture monitor start unavailable')
            def is_alive(self):return False
            def join(self,**kwargs):raise AssertionError('unstarted monitor must not be joined')
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp)
            with patch.object(runner.paths,'RESULTS_DIR',root),patch.object(runner.threading,'Thread',UnstartedMonitor):
                result=runner.run(SimpleNamespace(out=root/'fresh-calibration',native_seconds=3600))
            self.assertEqual(result['status'],'error')
            self.assertIn('fixture monitor start unavailable',result['error'])
            self.assertEqual(result['processes'],[])
            self.assertEqual(len((root/'registry.jsonl').read_text().splitlines()),1)
            self.assertEqual(json.loads((root/'fresh-calibration/completion.json').read_text())['status'],'error')

    def test_one_context_repeated_cells_observable_by_final_sum(self):
        plan,manifest,metadata=fixture()
        self.assertEqual(len(plan['execution']),699)
        self.assertEqual(len(metadata['target_operations']),200)
        self.assertEqual(len(manifest['blobs']),9)
        self.assertEqual(len(plan['external_inputs']),1)
        self.assertEqual(len(plan['final_outputs']),1)
        counts=Counter(t['cell_index']for t in metadata['target_operations'])
        self.assertEqual(Counter(counts.values()),{5:38,10:1})
        self.assertEqual(sum(op['op']=='boot'for op in plan['execution']),10)
        self.assertEqual(len({v['id']for v in plan['values']}),len(plan['values']))
        targets={t['target_output']for t in metadata['target_operations']}
        ancestors=set(plan['final_outputs'])
        for op in reversed(plan['execution']):
            if op['output']in ancestors:ancestors.update(op['inputs'])
        self.assertTrue(targets<=ancestors)

    def test_public_replay_matches_independent_repeated_direct_formulas(self):
        from array import array
        plan,_,metadata=fixture();packed=deterministic_fixture(32)
        constants={row['content']:array('d',[row['value']])*32 for row in metadata['constant_vectors']}
        replay,_=replay_profile(plan,constants,packed)
        direct=direct_profile_sum(metadata,packed,precision_bits=50)
        self.assertTrue(compare_references(direct,replay)['passed'])

    def test_new_samples_produce_five_ten_count_mean_and_sample_std(self):
        _,_,metadata=fixture();summary=own.summarize(dict(calibration_observations=observations(metadata)),metadata)
        self.assertEqual(summary['target_count'],200)
        for row in summary['cells']:
            count=10 if row['cell']['op']=='boot'else 5
            self.assertEqual(row['repetitions'],count)
            self.assertEqual(row['statistics']['operation_ns']['mean'],2000+(count-1)/2)
            self.assertGreater(row['statistics']['operation_ns']['sample_std'],0)
        self.assertFalse(summary['old_durations_reused'])

    def test_missing_duplicate_failed_or_invalid_new_observations_rejected(self):
        _,_,metadata=fixture();base=observations(metadata)
        cases=[base[:-1],base[:-1]+[base[0]],copy.deepcopy(base),copy.deepcopy(base),copy.deepcopy(base)]
        cases[2][0]['completed']=False;cases[3][0]['operation_ns']=0
        cases[4][0]['compute_ns']=3000
        for rows in cases:
            with self.assertRaises(ValueError):own.summarize(dict(calibration_observations=rows),metadata)

    def test_projection_covers_metadata_and_keeps_full_wall_unmeasured(self):
        _,_,metadata=fixture();summary=own.summarize(dict(calibration_observations=observations(metadata)),metadata)
        source=make_catalog_source(metadata['cells'],c2s_bits=52)
        projection=own.project(source,summary)
        self.assertEqual(projection['status'],'complete_operator_projection')
        self.assertEqual(projection['execution_count'],39)
        self.assertGreaterEqual(projection['major90_covered_fraction'],.9)
        self.assertIsNone(projection['full_inference_prediction_s'])
        self.assertFalse(projection['fit_to_whole_model_clock'])
        summary['cells']=summary['cells'][:-1]
        partial=own.project(source,summary)
        self.assertEqual(partial['status'],'partial_operator_projection')
        self.assertIsNone(partial['operator_component_prediction_s'])
        self.assertEqual(len(partial['missing_cells']),1)

    def test_native_argv_explicit_targets_and_same_public_parameters(self):
        source=dict(plan='/plan',bundle='/bundle',input='/input',expected='/expected',metadata='/targets')
        argv=runner.native_argv(Path('/client'),source,Path('/out'),'execute')
        self.assertEqual(argv[argv.index('-calibration-metadata')+1],'/targets')
        for name,value in(('-bootstrap-cache','32'),('-precision-bits','50'),('-c2s-bits','52'),
            ('-auxiliary-prime-bits','61'),('-mod1-inv-degree','15'),('-log-message-ratio','2')):
            self.assertEqual(argv[argv.index(name)+1],value)
        self.assertNotIn('-trace-key-requests',argv)
        self.assertNotIn('-calibration-metadata',runner.native_argv(Path('/client'),source,Path('/out'),'parameters'))

    def test_private_source_records_only_in_memory_and_captures_complete_operation(self):
        source=own.build_source()
        self.assertIn('calibrationOperationBegin=time.Now()',source)
        self.assertIn('if name=="runtime.compute" && preparedTimed {return calibrationStage(metadata,fn)}',source)
        self.assertIn('calibrationSamples=append(calibrationSamples',source)
        self.assertIn('calibration_persistence_during_inference',source)
        self.assertIn('calibrationPublish()',source)
        fragment=Path(own.__file__).with_suffix('.go').read_text()
        self.assertNotIn('save(',fragment);self.assertNotIn('emit(',fragment)
        self.assertNotIn('Encrypt',fragment);self.assertNotIn('Encode',fragment)


_GOFMT=paths.WORKSPACE/'baselines/heir-r20-go-backend-20261115/toolchain/go/bin/gofmt'


@require(_GOFMT)
class CalibrationGoSyntaxTests(unittest.TestCase):
    def test_generated_go_parses_without_build_or_native_execution(self):
        result=subprocess.run([str(_GOFMT),'-e'],input=own.build_source(),text=True,
            capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
