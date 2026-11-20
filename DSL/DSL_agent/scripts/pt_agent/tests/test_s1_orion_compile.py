"""New tiny dispatch, metadata and terminal fixtures; no Orion/model calls."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import s1_orion_compile as outer
from pt_agent.experiments import cold_rule63_whole_model as cold
from pt_agent.experiments import dacapo_compile_continuation as resources


def config():
    return dict(ckks_params=dict(LogN=16, LogQ=[55]+[40]*5, LogScale=40),
                orion=dict(backend='lattigo', io_mode='none', margin=1.5))


def no_placement():
    return dict(compile_calls=1, official_exception_origin=True,
        exception_type='ValueError', function='finally_solve_full_level_dag',
        message='Automatic bootstrap placement failed. Actual official exception',
        predicted_before_compile=False, first_depth_exceeding_l_eff=dict(degree=63, depth=7))


class OrionOuterTest(unittest.TestCase):
    def test_worker_normalization_and_actual_exception_category(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'phase_result.json'
            path.write_text(json.dumps(dict(mode='orion-native-rule63-fit-compile',
                status='no_placement', error=dict(type='ValueError', message='actual'),
                compiled_execution_bootstraps=None)))
            value = outer.read(path)
            self.assertEqual(value['error'], 'ValueError: actual')
            self.assertEqual(value['native_error'], dict(type='ValueError', message='actual'))
            self.assertIsNone(value['compiled_bootstraps'])
            result = dict(status='error', phases=dict(compile=dict(status='no_placement',
                compile_calls=1, no_placement_proof=no_placement())), processes=[])
            self.assertEqual(outer.classification(result), 'no_placement')
            result['phases']['compile']['no_placement_proof']['predicted_before_compile'] = True
            self.assertEqual(outer.classification(result), 'driver_error')
            result['guard_error'] = 'resource_limit: fixture'
            self.assertEqual(outer.classification(result), 'resource_limit')

    def test_prepared_config_and_ten_explicit_worker_flags(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source'; source.mkdir()
            selected = {k:str(root/k) for k in ('checkpoint','model_source','model_info','records','selections')}
            (source/'source-proof.json').write_text(json.dumps(dict(selected=selected)))
            config_path = root/'prepared.json'; config_path.write_text(json.dumps(config()))
            self.assertEqual(outer.configuration(config_path), config())
            args = SimpleNamespace(native_seconds=7200, out=root, short_model='resnet20',
                orion_root=root/'orion', orion_python=root/'orion-venv/bin/python')
            command, timeout = outer.phase_command(args, 'compile', 7400)
            self.assertEqual(timeout, 7200)
            self.assertEqual(command[:4], [str(args.orion_python), '-B', '-m', outer.WORKER])
            pairs = dict(zip(command[4::2], command[5::2]))
            self.assertEqual(set(pairs), {'--model','--work-dir','--checkpoint','--model-source',
                '--model-info','--records','--selections','--orion-root','--config'})
            self.assertEqual(pairs['--records'], selected['records'])
            self.assertEqual(pairs['--model'], 'resnet20')
            self.assertNotIn('nixrun', command)
            bad = config(); bad['ckks_params']['LogScale'] = 43
            config_path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'Prepared native'):
                outer.configuration(config_path)

    def test_paper_preserves_actual_native_clocks_and_parameter_difference(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(out=root, short_model='resnet20', source_case=root/'a2b',
                                   compatibility_case=root/'r16')
            worker = dict(status='no_placement', error='actual official failure', compile_calls=1,
                fit_calls=1, no_placement_proof=no_placement(), fit_compile_seconds=3.,
                init_scheme_seconds=8., stage_clocks_ns=dict(fit=dict(elapsed_ns=1000000000),
                compile=dict(elapsed_ns=2000000000)), compiled_execution_bootstraps=None)
            result = dict(status='error', error='outer observation', phases=dict(compile=worker), processes=[])
            row = outer.paper_record(args, result, dict(complete_case_elapsed_seconds=12.), {})
            self.assertEqual(row['status'], 'no_placement')
            self.assertEqual((row['compile_s'],row['init_scheme_seconds'],row['actual_case_seconds']), (3.,8.,12.))
            self.assertIsNone(row['generation_e2e_s'])
            self.assertIsNone(row['boots_compiled'])
            self.assertIsNone(row['pred_latency_s'])
            self.assertIsNone(row['quality']['value'])
            self.assertFalse(row['same_full_parameters'])
            self.assertFalse(row['domain_equivalent'])
            self.assertEqual(row['native_parameter_mapping']['official_input_level'], 5)
            self.assertEqual(row['native_parameter_mapping']['comparison_input_level'], 8)

    def test_reused_supervisor_closes_fake_success_and_no_placement(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = dict(split='train', count=1000, batch=1, indices=list(range(1000)),
                sample_ids=[str(i) for i in range(1000)], normalization={})
            for status in ('ok', 'no_placement'):
                out = root/status
                args = SimpleNamespace(out=out, model='resnet20_silu', short_model='resnet20',
                    placement='orion', source_case=root/'a2b', compatibility_case=root/'r16',
                    case_gib=32, seconds=240, native_seconds=7200, paper_runs=root/'runs.jsonl',
                    orion_root=root/'orion', orion_python=root/'orion-venv/bin/python')
                proof = dict(calibration=calibration, selected={k:str(root/k) for k in
                    ('checkpoint','model_source','model_info','records','selections')})
                def snapshot(current, evidence):
                    folder = current.out/'source'; folder.mkdir()
                    (folder/'source-proof.json').write_text(json.dumps(evidence))
                    (folder/'orion-config.json').write_text(json.dumps(config()))
                def fake_process(command, **kwargs):
                    self.assertEqual(kwargs['mem_gib'],24)
                    self.assertEqual(kwargs['fsize_gib'],16)
                    self.assertNotIn('LD_LIBRARY_PATH', kwargs['env'])
                    self.assertIn('PYTHONPATH', kwargs['env'])
                    start = time.perf_counter_ns()
                    payload = dict(mode='orion-native-rule63-fit-compile', status=status,
                        compiled_execution_bootstraps=None, native_compile_accepted=status=='ok',
                        fit_calls=1, fit_completed=1, compile_calls=1, compile_completed=int(status=='ok'),
                        init_scheme_calls=1, init_scheme_completed=1, calibration_samples=1000,
                        calibration_source=calibration, strict_translation_math_passed=True,
                        actual_CKKS_executions=0, fit_compile_seconds=.01, init_scheme_seconds=.02,
                        error=None if status=='ok' else 'actual official exception',
                        no_placement_proof=no_placement() if status!='ok' else None)
                    (out/'work/compile/phase_result.json').write_text(json.dumps(payload))
                    return dict(exit_code=0 if status=='ok' else 1, timed_out=False,
                        signal=None, remaining_pids=[], signals_sent=[], started_ns=start,
                        ended_ns=time.perf_counter_ns())
                with patch.object(outer.paths,'RESULTS_DIR',root), \
                     patch.object(outer,'source_gate',return_value=proof), \
                     patch.object(outer,'snapshot',side_effect=snapshot), \
                     patch.object(cold,'run_process',side_effect=fake_process), \
                     patch.object(resources,'available_ram',return_value=16*outer.GIB):
                    result = outer.execute(args)
                self.assertEqual(result['status'], 'ok' if status=='ok' else 'error')
                self.assertEqual(result['attempted'],dict(compile=1))
                self.assertEqual(result['completed'],dict(compile=1) if status=='ok' else {})
                completion = json.loads((out/'completion.json').read_text())
                self.assertTrue(completion['registry_persisted'])
                row = json.loads((out/'paper-record.json').read_text())
                self.assertEqual(row['status'],status)
                registry = [json.loads(line) for line in (root/'registry.jsonl').read_text().splitlines()
                            if json.loads(line)['run_id'] == out.name]
                self.assertEqual(len(registry),1)
                self.assertEqual(set(registry[0]),outer.closed.REGISTRY_FIELDS)


if __name__ == '__main__':
    unittest.main()
