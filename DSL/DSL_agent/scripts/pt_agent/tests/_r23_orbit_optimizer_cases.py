"""R23 observer fixtures; all solver, optimizer and termination objects are fake."""

import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class SimulatedTermination(BaseException):
    pass


class FakeParams:
    def __init__(self, profile, sysname, mode, **kwargs):
        self.constructor = (profile, sysname, mode, dict(kwargs))
        self.poly_deg, self.max_slot = 32768, 16384
        self.bts_lb, self.bts_ub, self.lvl_lb, self.lvl_ub = 3, 16, 1, 16
        self.Sf, self.backend = 51, 'Lattigo'
        self.mode, self.sysname = mode, sysname
        self.Sw, self.Csw = kwargs['Sw'], kwargs['CSw']
        for name in ('bpsdepth', 'threads', 'comp', 'part', 'reqbp', 'netname'):
            setattr(self, name, kwargs[name])
        self.dacapo_mlir_in = self.dacapo_mlir_out = True
        self.trunc_val = 1
        self.latency_table = {'fixture':123}


class FakePartition:
    name = 'fixture_partition'
    inputs, outputs = {'input'}, {'output'}
    nodes = ['input', 'middle', 'output']
    edges = [('input', 'middle'), ('middle', 'output')]
    def number_of_nodes(self):return 3
    def number_of_edges(self):return 2


class FakeGraph:
    def __init__(self, record):
        self.record = record
        self.inputs, self.outputs = set(record['inputs']), set(record['outputs'])
    def nodes(self, data=False):
        return [(row['id'], {k:v for k,v in row.items() if k != 'id'})
                for row in self.record['nodes']]
    def edges(self, data=False):
        return [(row['source'], row['target'], {k:v for k,v in row.items()
                if k not in ('source','target')}) for row in self.record['edges']]


def fake_ilp_module(*, response=None, error=None):
    module = ModuleType('r23_fake_ilp_worker')
    module.solve_ilp = Mock(return_value=response, side_effect=error)
    module.traceback = SimpleNamespace(print_exc=Mock())
    class ILP_Worker:
        def _lsabts_worker(self, *, assert_after=False):
            pdag = FakePartition()
            io_budget = dict(in_lvl=16, in_scl=40, out_lvl=1)
            task_name = 'Partition_fixture_in_lvl=16_in_scl=40_out_lvl=1'
            try:
                returned = module.solve_ilp(pdag, io_budget, 'fake-estimator', task_name, 1, 'fake-params')
                if assert_after:
                    raise AssertionError('fixed assertion after solve')
                return returned
            except Exception:
                module.traceback.print_exc()
            raise AssertionError('A fatal observer must stop the fake queue')
    module.ILP_Worker = ILP_Worker
    return module


def graph_fixture():
    return dict(inputs=['actual_input'], outputs=['actual_output'],
        nodes=[dict(id='actual_input', op='input', level=16, scale=40, weight=1, op_descr={}),
               dict(id='placed_boot', op='bootstrap', level=16, scale=51, weight=7,
                    op_descr=dict(targetLevel=13)),
               dict(id='actual_output', op='add', level=1, scale=40, weight=2, op_descr={})],
        edges=[dict(source='actual_input', target='placed_boot', weight=1),
               dict(source='placed_boot', target='actual_output', weight=1)])


class R23OrbitOptimizerCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pt_agent.experiments import r23_orbit_optimizer_worker as worker
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        cls.worker, cls.Recorder = worker, StageRecorder

    def setUp(self):
        temporary = TemporaryDirectory(prefix='r23-inner-', dir=os.environ.get('R23_TEST_WORK'))
        self.addCleanup(temporary.cleanup)
        self.case = Path(temporary.name)
        self.work = self.case / 'work/optimize-worker'; self.work.mkdir(parents=True)
        self.recorder = self.Recorder(self.work / 'events.jsonl')
        self.profile = self.case / 'orbit/cost_models/profiled_LATTIGONEW_CPU64k_3_16.json'
        self.profile.parent.mkdir(parents=True)
        self.profile.write_text(json.dumps(dict(polynomialDegree=131072, rescalingFactor=51,
            levelLowerBound=1, levelUpperBound=16, bootstrapLevelLowerBound=3,
            bootstrapLevelUpperBound=16, runtime='Lattigo')))
        license_file = self.case / 'orbit-venv/lib/python3.11/site-packages/gurobipy/.libs/gurobi.lic'
        license_file.parent.mkdir(parents=True); license_file.write_text('metadata fixture only\n')
        input_file = self.case / 'r22/trace/original.mlir'
        input_file.parent.mkdir(parents=True); input_file.write_text('fake source, never parsed\n')
        self.args = SimpleNamespace(phase='optimize', work_dir=self.work,
            orbit_root=self.case / 'orbit', profile=self.profile, license_file=license_file,
            input_mlir=input_file, output_mlir=self.work.parent / 'optimized.mlir',
            source_record=self.case / 'source-record.json')

    def _solve(self, module, observer, **kwargs):
        partition, budget, estimator, params = FakePartition(), dict(in_lvl=16), object(), object()
        arguments = (partition, budget, estimator, 'fake_task', 1, params)
        return module.solve_ilp(*arguments, **kwargs), arguments

    def test_params_preserve_original_native_constructor_and_in_memory_degree_alias(self):
        for mode in ('compile', 'execute'):
            with self.subTest(mode=mode):
                params, metadata = self.worker.make_params(self.profile, FakeParams, mode=mode)
                self.assertEqual(params.constructor, (str(self.profile), 'Orbit', mode,
                    dict(Sw=40, CSw=40, bpsdepth=15, threads=1, comp=True, part=True,
                         reqbp=False, netname='ResNetSiLU64k')))
                self.assertEqual((params.poly_deg, params.max_slot, params.Sf), (131072, 65536, 51))
                self.assertEqual(params.latency_table, {'fixture':123})
                self.assertTrue(params.dacapo_mlir_in and params.dacapo_mlir_out)
                self.assertEqual(params.trunc_val, 1)
                self.assertIn('32768', json.dumps(metadata))

    def test_params_reject_changed_degree_and_original_native_bounds(self):
        for key, value in (('polynomialDegree', 32768), ('rescalingFactor', 50),
                           ('levelUpperBound', 15), ('bootstrapLevelLowerBound', 2)):
            def altered(profile, sysname, mode, **kwargs):
                params = FakeParams(profile, sysname, mode, **kwargs)
                if key == 'rescalingFactor':params.Sf = value
                elif key == 'levelUpperBound':params.lvl_ub = value
                elif key == 'bootstrapLevelLowerBound':params.bts_lb = value
                return params
            original = json.loads(self.profile.read_text())
            changed = dict(original); changed[key] = value
            self.profile.write_text(json.dumps(changed))
            with self.subTest(key=key), self.assertRaises((ValueError, RuntimeError)):
                self.worker.make_params(self.profile, altered, mode='compile')
            self.profile.write_text(json.dumps(original))

    def test_solve_return_object_arguments_and_global_traceback_are_preserved(self):
        import traceback
        original_print = traceback.print_exc
        expected, marker = (object(), object()), object()
        module = fake_ilp_module(response=expected)
        original_solve, original_traceback = module.solve_ilp, module.traceback
        with self.worker.install_observers(module, self.recorder, self.work) as observer:
            actual, arguments = self._solve(module, observer, marker=marker)
            self.assertIs(actual, expected)
            original_solve.assert_called_once_with(*arguments, marker=marker)
            self.assertIs(traceback.print_exc, original_print)
            self.assertEqual(observer['counters'], dict(solve_ilp_attempts=1, solve_ilp_completed=1,
                solve_ilp_feasible=1, solve_ilp_infeasible=0, solve_ilp_errors=0))
        self.assertIs(module.solve_ilp, original_solve)
        self.assertIs(module.traceback, original_traceback)

    def test_normal_infeasible_return_is_preserved_without_termination(self):
        module = fake_ilp_module(response=(None, None))
        with patch.object(self.worker, '_terminate_self') as terminate:
            with self.worker.install_observers(module, self.recorder, self.work) as observer:
                returned, _ = self._solve(module, observer)
                self.assertEqual(returned, (None, None))
                self.assertEqual(observer['counters']['solve_ilp_infeasible'], 1)
                self.assertEqual(observer['counters']['solve_ilp_completed'], 1)
            terminate.assert_not_called()
        self.assertFalse((self.work / 'first-fatal.json').exists())

    def test_solve_exception_is_rethrown_with_original_identity(self):
        original = ValueError('fixed solver fixture error')
        module = fake_ilp_module(error=original)
        with patch.object(self.worker, '_terminate_self') as terminate:
            with self.worker.install_observers(module, self.recorder, self.work) as observer:
                with self.assertRaises(ValueError) as caught:self._solve(module, observer)
                self.assertIs(caught.exception, original)
                self.assertEqual(observer['counters']['solve_ilp_attempts'], 1)
                self.assertEqual(observer['counters']['solve_ilp_completed'], 0)
                self.assertEqual(observer['counters']['solve_ilp_errors'], 1)
            terminate.assert_not_called()

    def test_worker_caught_solve_exception_saves_first_fatal_then_terminates(self):
        module = fake_ilp_module(error=ValueError('fixed caught solver error'))
        original_print = module.traceback.print_exc
        with patch.object(self.worker, '_terminate_self', side_effect=SimulatedTermination) as terminate:
            with self.assertRaises(SimulatedTermination):
                with self.worker.install_observers(module, self.recorder, self.work):
                    module.ILP_Worker()._lsabts_worker()
        terminate.assert_called_once_with()
        original_print.assert_called_once_with()
        fatal = json.loads((self.work / 'first-fatal.json').read_text())
        self.assertEqual(fatal['termination_origin'], 'worker_caught_exception')
        self.assertIn('fixed caught solver error', json.dumps(fatal))
        self.assertIn('Partition_fixture', json.dumps(fatal))
        self.assertEqual(fatal['counters']['solve_ilp_errors'], 1)
        self.assertEqual(fatal['counters']['solve_ilp_completed'], 0)
        events = [json.loads(line) for line in (self.work / 'solve-ilp-events.jsonl').read_text().splitlines()]
        self.assertEqual([item['type'] for item in events], ['begin', 'error'])

    def test_worker_post_solve_assertion_is_recorded_after_a_completed_solve(self):
        module = fake_ilp_module(response=(object(), 2.0))
        with patch.object(self.worker, '_terminate_self', side_effect=SimulatedTermination):
            with self.assertRaises(SimulatedTermination):
                with self.worker.install_observers(module, self.recorder, self.work):
                    module.ILP_Worker()._lsabts_worker(assert_after=True)
        fatal = json.loads((self.work / 'first-fatal.json').read_text())
        self.assertIn('fixed assertion after solve', json.dumps(fatal))
        self.assertEqual(fatal['termination_origin'], 'worker_caught_exception')
        self.assertEqual(fatal['counters']['solve_ilp_completed'], 1)
        self.assertEqual(fatal['counters']['solve_ilp_errors'], 0)
        events = [json.loads(line) for line in (self.work / 'solve-ilp-events.jsonl').read_text().splitlines()]
        self.assertEqual([item['type'] for item in events], ['begin', 'end'])

    def test_fatal_notification_write_failure_still_terminates(self):
        module = fake_ilp_module(error=RuntimeError('fixed error before record failure'))
        original_write = self.worker.write_json
        def write(path, record):
            if Path(path).name == 'first-fatal.json':raise OSError('fixed fatal write failure')
            return original_write(path, record)
        with patch.object(self.worker, '_terminate_self', side_effect=SimulatedTermination) as terminate, \
                patch.object(self.worker, 'write_json', side_effect=write):
            with self.assertRaises(SimulatedTermination):
                with self.worker.install_observers(module, self.recorder, self.work):
                    module.ILP_Worker()._lsabts_worker()
        terminate.assert_called_once_with()

    def test_fatal_original_print_failure_still_terminates(self):
        module = fake_ilp_module(error=RuntimeError('fixed error before print failure'))
        module.traceback.print_exc.side_effect = RuntimeError('fixed print failure')
        with patch.object(self.worker, '_terminate_self', side_effect=SimulatedTermination) as terminate:
            with self.assertRaises(SimulatedTermination):
                with self.worker.install_observers(module, self.recorder, self.work):
                    module.ILP_Worker()._lsabts_worker()
        terminate.assert_called_once_with()

    def test_nonworker_traceback_caller_delegates_without_termination(self):
        module = fake_ilp_module(response=(None, None))
        original = module.traceback.print_exc
        with patch.object(self.worker, '_terminate_self') as terminate:
            with self.worker.install_observers(module, self.recorder, self.work):
                try:raise ValueError('unrelated fake caller')
                except ValueError:module.traceback.print_exc()
        original.assert_called_once_with()
        terminate.assert_not_called()
        self.assertFalse((self.work / 'first-fatal.json').exists())

    def test_placement_records_bootstrap_nodes_weights_and_original_level_scale(self):
        graph = graph_fixture(); before = copy.deepcopy(graph)
        placement = self.worker._placement_record(graph)
        self.assertEqual(graph, before)
        self.assertEqual(placement['placement_bootstrap_nodes'], 1)
        self.assertEqual(placement['placement_bootstrap_weight_sum'], 7)
        self.assertIn('placed_boot', json.dumps(placement))
        self.assertIn('targetLevel', json.dumps(placement))

    def test_placement_rejects_unknown_output_cycle_and_unreachable_graph(self):
        variants = []
        graph = graph_fixture(); graph['outputs'] = ['unknown']; variants.append(graph)
        graph = graph_fixture(); graph['edges'].append(dict(source='actual_output', target='actual_input', weight=1)); variants.append(graph)
        graph = graph_fixture(); graph['edges'].pop(); variants.append(graph)
        graph = graph_fixture(); graph['nodes'].append(copy.deepcopy(graph['nodes'][0])); variants.append(graph)
        for graph in variants:
            with self.subTest(graph=graph), self.assertRaises((ValueError, RuntimeError)):
                self.worker._placement_record(graph)

    def test_environment_sets_fixed_license_before_import_and_restores_all_process_settings(self):
        result = self.worker._initial_result('optimize')
        with patch.dict(os.environ, GRB_LICENSE_FILE='old-secret', GRB_OTHER='old-secret',
                        GUROBI_HOME='old-secret', HOME='/fixture-home'):
            before_env, before_path = dict(os.environ), list(sys.path)
            before_cwd, before_signal = Path.cwd(), signal.getsignal(signal.SIGTERM)
            with self.worker._worker_environment(self.args, result):
                self.assertEqual(os.environ['GRB_LICENSE_FILE'], str(self.args.license_file))
                self.assertNotIn('GRB_OTHER', os.environ)
                self.assertNotIn('GUROBI_HOME', os.environ)
                self.assertEqual(os.environ['HOME'], '/fixture-home')
                self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)
                self.assertEqual(Path.cwd(), self.work)
                self.assertEqual(sys.path[0], str(self.args.orbit_root))
                sys.modules['r23_fixture_temporary_module'] = ModuleType('r23_fixture_temporary_module')
                for name in ('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','XDG_CONFIG_HOME','MPLCONFIGDIR'):
                    self.assertTrue(Path(os.environ[name]).is_relative_to(self.work))
            self.assertEqual(dict(os.environ), before_env)
            self.assertEqual(sys.path, before_path)
            self.assertEqual(Path.cwd(), before_cwd)
            self.assertEqual(signal.getsignal(signal.SIGTERM), before_signal)
            self.assertNotIn('r23_fixture_temporary_module', sys.modules)
        self.assertTrue(result['environment_restored'])
        self.assertNotIn('old-secret', json.dumps(result))

    def test_optimizer_seed_and_run_delegate_once_and_preserve_solver_counts(self):
        module = fake_ilp_module(response=(None, None))
        def run(input_file, output_file, params):
            self.assertEqual((input_file, output_file), (str(self.args.input_mlir), str(self.args.output_mlir)))
            module.solve_ilp(FakePartition(), {'in_lvl':16}, object(), 'single_task', 1, params)
            Path(output_file).write_text('fake optimized output\n')
        optimizer = SimpleNamespace(Params=FakeParams, set_global_seed=Mock(), run=Mock(side_effect=run))
        result = self.worker._initial_result('optimize')
        with patch.object(self.worker, '_check_r22_source', return_value={'fixture':True}), \
                patch.object(self.worker, '_load_optimizer', return_value=(optimizer,module)):
            self.worker.optimize_model(self.args, self.recorder, result)
        optimizer.set_global_seed.assert_called_once_with(42)
        optimizer.run.assert_called_once()
        self.assertTrue(result['native_optimizer_completed'])
        self.assertEqual((result['optimize_attempts'],result['optimize_completed']), (1,1))
        self.assertEqual((result['solve_ilp_attempts'],result['solve_ilp_completed'],result['solve_ilp_infeasible']), (1,1,1))
        self.assertTrue(result['observer_restored'])
        self.assertIsNone(result['model_optimize_calls'])
        self.assertTrue(all(result[name] == 0 for name in self.worker.ZERO_FIELDS))

    def test_optimizer_failure_preserves_partial_counts_and_original_module_references(self):
        module = fake_ilp_module(error=ValueError('fixed fake solve failure'))
        original_solve, original_traceback = module.solve_ilp, module.traceback
        def run(input_file, output_file, params):
            module.solve_ilp(FakePartition(), {}, object(), 'failed_task', 1, params)
        optimizer = SimpleNamespace(Params=FakeParams, set_global_seed=Mock(), run=Mock(side_effect=run))
        result = self.worker._initial_result('optimize')
        with patch.object(self.worker, '_check_r22_source', return_value={'fixture':True}), \
                patch.object(self.worker, '_load_optimizer', return_value=(optimizer,module)), \
                self.assertRaisesRegex(ValueError, 'fixed fake solve failure'):
            self.worker.optimize_model(self.args, self.recorder, result)
        self.assertEqual((result['optimize_attempts'],result['optimize_completed']), (1,0))
        self.assertEqual((result['solve_ilp_attempts'],result['solve_ilp_completed'],result['solve_ilp_errors']), (1,0,1))
        self.assertIs(module.solve_ilp, original_solve)
        self.assertIs(module.traceback, original_traceback)
        self.assertTrue(result['observer_restored'])

    def test_validation_parses_checks_and_saves_assigned_graph_once(self):
        self.args.output_mlir.write_text('fake output fixture\n')
        builder, checker = Mock(return_value=FakeGraph(graph_fixture())), Mock()
        result = self.worker._initial_result('validate')
        with patch.object(self.worker, '_validate_source', return_value={'fixture':True}), \
                patch.object(self.worker, '_load_validation', return_value=(FakeParams,builder,checker)):
            self.worker.validate_model(self.args, self.recorder, result)
        builder.assert_called_once()
        checker.assert_called_once_with(builder.return_value)
        self.assertEqual(builder.call_args.args[1].mode, 'execute')
        self.assertEqual((result['validate_attempts'],result['validate_completed']), (1,1))
        self.assertEqual((result['build_from_mlir_calls'],result['check_tdag_calls']), (1,1))
        self.assertTrue(result['placement_artifact_checked'])
        self.assertEqual((result['placement_bootstrap_nodes'],result['placement_bootstrap_weight_sum']), (1,7))
        self.assertEqual(json.loads((self.work/'tdag.json').read_text()), graph_fixture())

    def test_validation_original_assertion_failure_keeps_raw_graph_and_partial_counts(self):
        self.args.output_mlir.write_text('fake output fixture\n')
        builder = Mock(return_value=FakeGraph(graph_fixture()))
        checker = Mock(side_effect=AssertionError('fixed official checker failure'))
        result = self.worker._initial_result('validate')
        with patch.object(self.worker, '_validate_source', return_value={'fixture':True}), \
                patch.object(self.worker, '_load_validation', return_value=(FakeParams,builder,checker)), \
                self.assertRaisesRegex(AssertionError, 'fixed official checker failure'):
            self.worker.validate_model(self.args, self.recorder, result)
        self.assertTrue((self.work/'tdag.json').is_file())
        self.assertEqual((result['check_tdag_calls'],result['check_tdag_completed']), (1,0))
        self.assertEqual((result['validate_attempts'],result['validate_completed']), (1,0))
        self.assertFalse(result['placement_artifact_checked'])


if __name__ == '__main__':
    unittest.main()
