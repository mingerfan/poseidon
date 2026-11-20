from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import a4_calibration_delivery as own


def fixture(root,*,calibration_status='ok',native_status='ok'):
    registry=root/'registry.jsonl';rows=[]
    calibration=root/'public-calibration';native=root/'actual-native'
    plan=root/'original123-plan.json'
    summary=dict(status='ok',cell_count=39,target_count=200,old_durations_reused=False,
        parameter_group='r31-N16-C2S52-P61-scale50',cells=[dict(index=i,repetitions=10 if i==0 else 5)
            for i in range(39)],cost_scope='Independent public operator samples with prepared disk reads')
    projection=dict(status='complete_operator_projection',execution_count=17863,bootstrap_count=123,
        old_durations_reused=False,fit_to_whole_model_clock=False,full_inference_prediction_s=None,
        operator_component_prediction_s=20,source_plan=str(plan),weighted_cells=[dict(cell='fixture',source_count=3)],
        scope='Operator spans including prepared disk loads',unmeasured_components=['Encryption','Loop gaps'])
    native_record=dict(status=native_status,error=None if native_status=='ok'else 'fixture native failure',
        source=dict(plan=str(plan)),ckks_inference_s=40 if native_status=='ok'else None,
        comparison=dict(passed=True,bootstrap_count=123,execution_count=17863),
        client_summary=dict(prepared_inference_scope='Encrypt through decrypt with disk loading',
            ckks_inference_s=40,ckks_inference_completed=True,
            no_keygen_or_constant_encoding_during_inference=True))
    for case,record in((calibration,dict(status=calibration_status,
        error=None if calibration_status=='ok'else 'fixture calibration failure')),(native,native_record)):
        case.mkdir()
        (case/'result.json').write_text(json.dumps(record))
        (case/'completion.json').write_text(json.dumps(dict(status=record['status'],
            registry_persisted=True,required_artifacts_persisted=True)))
        rows.append(dict(run_id=case.name,round='fixture',model='fixture',mode='fixture',git_rev=None,
            metrics=dict(status=record['status']),artifacts_dir=str(case),ts='2026-11-17T00:00:00+00:00'))
    if calibration_status=='ok':
        (calibration/'calibration-summary.json').write_text(json.dumps(summary))
        (calibration/'operator-projection.json').write_text(json.dumps(projection))
    registry.write_text(''.join(json.dumps(row)+'\n'for row in rows))
    return calibration,native,registry,summary,projection


class CalibrationDeliveryTests(unittest.TestCase):
    def test_finite_metadata_parser_rejects_nonfinite_values(self):
        for text in('{"x":NaN}','{"x":Infinity}','{"x":1e999}'):
            with self.assertRaises(ValueError):own.parse(text)

    def test_case_requires_unique_durable_registry_and_keeps_pending_distinct(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);calibration,_,registry,_,_=fixture(root)
            closed=own.inspect_case(calibration,registry)
            self.assertEqual(closed['state'],'closed')
            (calibration/'completion.json').unlink()
            self.assertEqual(own.inspect_case(calibration,registry)['state'],'pending')
            (calibration/'completion.json').write_text(json.dumps(dict(status='ok',
                registry_persisted=True,required_artifacts_persisted=True)))
            with registry.open('a')as stream:stream.write(json.dumps(closed['registry_record'])+'\n')
            with self.assertRaises(ValueError):own.inspect_case(calibration,registry)

    def test_public_tables_remain_unchanged_and_ratio_is_explicit_scope_diagnostic(self):
        with tempfile.TemporaryDirectory()as temp:
            calibration,native,registry,summary,projection=fixture(Path(temp))
            payloads=own.build_payloads(own.inspect_case(calibration,registry),own.inspect_case(native,registry))
            self.assertEqual(payloads['a4_prepared_disk_calibration']['data'],summary)
            self.assertEqual(payloads['a4_prepared_disk_operator_projection']['data'],projection)
            diagnostic=payloads['a4_prepared_disk_scope_diagnostic']
            self.assertEqual(diagnostic['native']['ckks_inference_s'],40)
            self.assertEqual(diagnostic['operator_projection_over_actual_ckks_ratio'],.5)
            self.assertEqual(diagnostic['ratio_label'],'Scope diagnostic only')
            self.assertIsNone(diagnostic['full_formal_prediction_s'])
            self.assertFalse(diagnostic['fit_to_whole_model_clock'])

    def test_closed_failures_publish_source_status_with_missing_values_preserved(self):
        with tempfile.TemporaryDirectory()as temp:
            calibration,native,registry,_,_=fixture(Path(temp),calibration_status='error',native_status='error')
            payloads=own.build_payloads(own.inspect_case(calibration,registry),own.inspect_case(native,registry))
            self.assertEqual(payloads['a4_prepared_disk_calibration']['status'],'driver_error')
            self.assertIsNone(payloads['a4_prepared_disk_calibration']['data'])
            diagnostic=payloads['a4_prepared_disk_scope_diagnostic']
            self.assertIsNone(diagnostic['native']['ckks_inference_s'])
            self.assertIsNone(diagnostic['operator_projection_over_actual_ckks_ratio'])

    def test_fresh_publication_uses_no_overwrite_and_no_output_readback(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp)
            with patch.object(Path,'read_text',side_effect=AssertionError('output readback forbidden')):
                published=own.publish_new(root,dict(one=dict(value=3),two=dict(value=None)))
                self.assertEqual(len(published),2)
                self.assertTrue(all(Path(path).is_file()for path in published))
                with self.assertRaises(ValueError):own.publish_new(root,dict(one=dict(value=4)))

    def test_closed_fixture_watch_publishes_once_without_sleep_or_benchmark_append(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);calibration,native,registry,_,_=fixture(root)
            args=SimpleNamespace(out=root/'watch',calibration=calibration,native=native,registry=registry,
                destination=root/'delivery',seconds=60,deadline=datetime.now(timezone.utc)+timedelta(seconds=60))
            with patch.object(own.time,'sleep',side_effect=AssertionError('closed source must not wait')):
                result=own.watch(args)
            self.assertEqual(result['status'],'ok');self.assertEqual(result['delivery_status'],'ok')
            self.assertEqual(result['polls'],1);self.assertEqual(len(result['published']),3)
            self.assertTrue(result['no_benchmark_runs_appended'])
            self.assertTrue((root/'watch/completion.json').is_file())
            self.assertFalse((root/'delivery/runs.jsonl').exists())

    def test_watch_expiry_delivers_missing_status_without_measurements(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);registry=root/'registry.jsonl';registry.touch()
            args=SimpleNamespace(out=root/'watch',calibration=root/'missing-calib',native=root/'missing-native',
                registry=registry,destination=root/'delivery',seconds=1,
                deadline=datetime.now(timezone.utc)+timedelta(seconds=60))
            captured={}
            def publish(destination,payloads):captured.update(payloads);return []
            with patch.object(own.time,'monotonic',side_effect=[0,0,1,2]), \
                 patch.object(own.time,'sleep'),patch.object(own,'publish_new',side_effect=publish):
                result=own.watch(args)
            self.assertEqual(result['delivery_status'],'timeout')
            self.assertIsNone(captured['a4_prepared_disk_calibration']['data'])
            self.assertIsNone(captured['a4_prepared_disk_scope_diagnostic']['native']['ckks_inference_s'])


if __name__=='__main__':unittest.main()
