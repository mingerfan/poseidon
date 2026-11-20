"""F5 deterministic fakes: no official trace, DSO call, model or optimizer."""

from contextlib import redirect_stderr
import copy
import io
import json
import os
from pathlib import Path
import sys
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pt_agent.experiments import r14_orbit_frontend as f5


def graph_fixture():
    return dict(inputs=['arg0', 'arg1'], outputs=['actual_add'],
                nodes=[dict(id=name, op=op, level=None, scale=None, weight=1, op_descr={})
                       for name, op in [('arg0', 'input'), ('arg1', 'input'),
                                        ('actual_mul', 'mul'), ('actual_add', 'add')]],
                edges=[dict(source=source, target=target, weight=1) for source, target in
                       [('arg0', 'actual_mul'), ('arg1', 'actual_mul'),
                        ('actual_mul', 'actual_add'), ('arg0', 'actual_add')]])


class FakeGraph:
    def __init__(self, record):
        self.record = record
        self.inputs = set(record['inputs'])
        self.outputs = set(record['outputs'])

    def nodes(self, data=False):
        return [(node['id'], {key: value for key, value in node.items() if key != 'id'})
                for node in self.record['nodes']]

    def edges(self, data=False):
        return [(edge['source'], edge['target'],
                 {key: value for key, value in edge.items() if key not in ('source', 'target')})
                for edge in self.record['edges']]


class Symbol:
    def __init__(self, text):
        self.text = text

    def __mul__(self, other):
        return Symbol('(' + self.text + '*' + other.text + ')')

    def __add__(self, other):
        return Symbol('(' + self.text + '+' + other.text + ')')


def fake_expr():
    module = SimpleNamespace(ctxt=object(), lt=SimpleNamespace(finalize=Mock()),
                             body_calls=0, registered=[])

    def func(signature):
        def decorate(body):
            module.registered.append((signature, body))
            # Calling the decorated object would enter the unused createCall path.
            return Mock(side_effect=AssertionError('The decorated function must not be called'))
        return decorate

    def save(*, dirs, cst_dirs):
        module.body_calls += 1
        signature, body = module.registered[0]
        if body(Symbol('a'), Symbol('b')).text != '((a*b)+a)':
            raise AssertionError('The expression body changed')
        target = Path(dirs) / 'r14_orbit_frontend.mlir'
        target.write_text('fixed protocol fixture\n')
        (Path(cst_dirs) / '_hecate_arithmetic.cst').write_text('fixture metadata only\n')
        return str(target)

    module.func, module.save = Mock(side_effect=func), Mock(side_effect=save)
    return module


class TraceChecks(unittest.TestCase):
    def run_trace(self, work, module=None, error=None):
        args = SimpleNamespace(phase='trace', work_dir=work, frontend=work / 'official-frontend')
        with patch.object(f5, '_load_expr', return_value=module, side_effect=error) as load, \
             patch.dict(os.environ), redirect_stderr(io.StringIO()):
            result = f5.run_phase(args)
            self.assertEqual(os.environ['HECATE'], str(args.frontend.resolve()))
        return result, load

    def test_one_registration_save_evaluation_and_finalize(self):
        with TemporaryDirectory() as temporary:
            work, module = Path(temporary), fake_expr()
            result, load = self.run_trace(work, module)
            self.assertEqual(result['status'], 'ok')
            load.assert_called_once_with((work / 'official-frontend').resolve())
            module.func.assert_called_once_with('c,c')
            module.save.assert_called_once_with(dirs=str(work / 'trace'), cst_dirs=str(work / 'constants'))
            module.lt.finalize.assert_called_once_with(module.ctxt)
            self.assertEqual(module.body_calls, 1)
            self.assertEqual(result['save_attempts'], 1)
            self.assertEqual(result['finalize_attempts'], 1)
            self.assertEqual(result['input_ir_elements'], 1)
            self.assertEqual(result['original_save_cleanup_passes'], f5.CLEANUP_PASSES)
            self.assertEqual(len(result['constant_files']), 1)

    def test_save_failure_still_finalizes_once_and_preserves_result(self):
        with TemporaryDirectory() as temporary:
            work, module = Path(temporary), fake_expr()
            module.save.side_effect = RuntimeError('fixed save failure')
            result, _ = self.run_trace(work, module)
            self.assertEqual(result['status'], 'error')
            module.lt.finalize.assert_called_once_with(module.ctxt)
            self.assertEqual(result['save_attempts'], 1)
            self.assertIn('fixed save failure', result['error'])
            self.assertEqual(json.loads((work / 'phase_result.json').read_text()), result)

    def test_module_loading_failure_has_no_save_or_finalize(self):
        with TemporaryDirectory() as temporary:
            result, _ = self.run_trace(Path(temporary), error=ImportError('missing local dependency'))
            self.assertEqual(result['status'], 'dependency_missing')
            self.assertFalse(result['expr_module_loaded'])
            self.assertEqual(result['save_attempts'], 0)
            self.assertEqual(result['finalize_attempts'], 0)

    def test_no_context_stops_before_registration_or_finalize(self):
        with TemporaryDirectory() as temporary:
            module = fake_expr()
            module.ctxt = None
            result, _ = self.run_trace(Path(temporary), module)
            self.assertEqual(result['status'], 'error')
            module.func.assert_not_called()
            module.save.assert_not_called()
            module.lt.finalize.assert_not_called()

    def test_save_path_must_be_nonempty_and_inside_trace(self):
        for empty in (False, True):
            with self.subTest(empty=empty), TemporaryDirectory() as temporary:
                work, module = Path(temporary), fake_expr()
                def save(*, dirs, cst_dirs):
                    path = (Path(dirs) if empty else work) / 'returned.mlir'
                    path.write_text('' if empty else 'outside trace')
                    return str(path)
                module.save.side_effect = save
                result, _ = self.run_trace(work, module)
                self.assertEqual(result['status'], 'artifact_invalid')
                module.lt.finalize.assert_called_once_with(module.ctxt)

    def test_file_loading_registers_fixed_module_before_exec_without_package_import(self):
        module = SimpleNamespace()
        loader = Mock()
        loader.exec_module.side_effect = lambda value: self.assertIs(sys.modules['orbit_expr_check'], value)
        spec = SimpleNamespace(loader=loader)
        with patch.object(f5.importlib.util, 'spec_from_file_location', return_value=spec) as from_file, \
             patch.object(f5.importlib.util, 'module_from_spec', return_value=module), \
             patch.dict(sys.modules), patch.object(f5.importlib, 'import_module') as package_import:
            self.assertIs(f5._load_expr(Path('/fixture/frontend')), module)
            from_file.assert_called_once_with('orbit_expr_check', Path('/fixture/frontend/python/hecate/hecate/expr.py'))
            loader.exec_module.assert_called_once_with(module)
            package_import.assert_not_called()


class ParseChecks(unittest.TestCase):
    def test_graph_acceptance_recovers_actual_operation_names_and_preserves_attributes(self):
        record = graph_fixture()
        self.assertEqual(f5.graph_record(FakeGraph(record)), record)
        checked = f5.check_graph(record)
        self.assertTrue(checked['passed'])
        self.assertEqual((checked['mul'], checked['output']), ('actual_mul', 'actual_add'))
        self.assertEqual((checked['nodes'], checked['edges']), (4, 4))

    def test_wrong_dependencies_counts_output_and_assigned_metadata_are_rejected(self):
        cases = []
        changed = graph_fixture(); changed['edges'][3]['source'] = 'arg1'; cases.append(changed)
        changed = graph_fixture(); changed['edges'].pop(); cases.append(changed)
        changed = graph_fixture(); changed['edges'].append(copy.deepcopy(changed['edges'][0])); cases.append(changed)
        changed = graph_fixture(); changed['outputs'] = ['actual_mul']; cases.append(changed)
        changed = graph_fixture(); changed['nodes'][2]['level'] = 1; cases.append(changed)
        changed = graph_fixture(); changed['nodes'][1].pop('scale'); cases.append(changed)
        changed = graph_fixture(); changed['edges'][0]['source'] = 'actual_add'; cases.append(changed)
        for index, record in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                f5.check_graph(record)

    def test_parser_calls_only_params_and_builder_once_with_explicit_metadata_profile(self):
        with TemporaryDirectory() as temporary:
            work = Path(temporary)
            args = SimpleNamespace(phase='parse', work_dir=work, orbit_root=work / 'orbit',
                                   profile=work / 'orbit' / f5.PROFILE, mlir=work / 'saved.mlir')
            constructor = Mock(return_value=SimpleNamespace(mode='compile', sysname='Orbit',
                                                            threads=1, max_slot=16384, poly_deg=32768))
            builder = Mock(return_value=FakeGraph(graph_fixture()))
            with patch.object(f5, '_load_parser', return_value=(constructor, builder)) as load:
                result = f5.run_phase(args)
            self.assertEqual(result['status'], 'ok')
            load.assert_called_once_with(args.orbit_root)
            constructor.assert_called_once_with(str(args.profile), 'Orbit', 'compile', threads=1)
            builder.assert_called_once_with(str(args.mlir), constructor.return_value)
            self.assertEqual(json.loads((work / 'tdag.json').read_text()), graph_fixture())
            for key in ('optimizer_runs', 'keygen_runs', 'real_ckks_attempts', 'llm_calls'):
                self.assertEqual(result[key], 0)

    def test_parse_acceptance_failure_retains_raw_graph(self):
        with TemporaryDirectory() as temporary:
            work = Path(temporary)
            record = graph_fixture()
            record['edges'].pop()
            args = SimpleNamespace(phase='parse', work_dir=work, orbit_root=work / 'orbit',
                                   profile=work / 'orbit' / f5.PROFILE, mlir=work / 'source.mlir')
            params = SimpleNamespace(mode='compile', sysname='Orbit', threads=1, max_slot=16384, poly_deg=32768)
            with patch.object(f5, '_load_parser', return_value=(Mock(return_value=params),
                                                               Mock(return_value=FakeGraph(record)))), \
                 redirect_stderr(io.StringIO()):
                result = f5.run_phase(args)
            self.assertEqual(result['status'], 'error')
            self.assertFalse(result['graph_check']['passed'])
            self.assertEqual(json.loads((work / 'tdag.json').read_text()), record)
            self.assertTrue((work / 'phase_result.json').is_file())

    def test_parser_imports_official_scripts_modules_from_explicit_root(self):
        root = Path('/fixture/orbit')
        params = SimpleNamespace(__file__=root / 'scripts/params/params.py', Params=Mock())
        parser = SimpleNamespace(__file__=root / 'scripts/tdag/mlir2tdag.py', build_from_mlir=Mock())
        previous = list(sys.path)
        try:
            with patch.object(f5.importlib, 'import_module', side_effect=[params, parser]) as load:
                self.assertEqual(f5._load_parser(root), (params.Params, parser.build_from_mlir))
            self.assertEqual(sys.path[0], str(root))
            self.assertEqual([call.args[0] for call in load.call_args_list],
                             ['scripts.params.params', 'scripts.tdag.mlir2tdag'])
        finally:
            sys.path[:] = previous


class OuterChecks(unittest.TestCase):
    def locations(self, root):
        locations = dict(results=root / 'results', repo=root, scripts=root / 'scripts',
                         orbit=root / 'orbit', frontend=root / 'orbit/frontend/dacapo',
                         intake_python=root / 'intake-python', orbit_python=root / 'orbit-venv/bin/python',
                         profile=root / 'orbit' / f5.PROFILE)
        locations['results'].mkdir()
        locations['scripts'].mkdir()
        for path in [locations['intake_python'], locations['orbit_python'], locations['profile'],
                     locations['frontend'] / 'python/hecate/hecate/expr.py',
                     locations['frontend'] / 'build/lib/libHecateFrontend.so',
                     locations['orbit'] / 'scripts/params/params.py',
                     locations['orbit'] / 'scripts/tdag/mlir2tdag.py']:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture\n')
        return locations

    def fake_process(self, argv, **kwargs):
        phase = argv[argv.index('--phase') + 1]
        work = Path(argv[argv.index('--work-dir') + 1])
        if phase == 'trace':
            target = work / 'trace/actual-returned.mlir'
            target.parent.mkdir()
            target.write_text('fixed protocol fixture\n')
            record = dict(status='ok', trace_runs=1, save_attempts=1, finalize_attempts=1,
                          mlir=dict(path=str(target), size_bytes=target.stat().st_size))
        else:
            record = dict(status='ok', parse_runs=1, graph_check=dict(passed=True))
        (work / 'phase_result.json').write_text(json.dumps(record))
        start = time.perf_counter_ns()
        event = dict(event_id='fixture-' + phase, stage_id='fixture.' + phase,
                     parent_id=None, start_ns=start, pid=10001 if phase == 'trace' else 10002,
                     ppid=0, metadata={}, cache_hit=False, artifact=None)
        (work / 'events.jsonl').write_text(json.dumps(dict(event, type='begin')) + '\n' +
                                         json.dumps(dict(event, type='end', end_ns=start + 1,
                                                         status='ok', error=None)) + '\n')
        Path(kwargs['log']).write_text('fake process; no native execution\n')
        return dict(exit_code=0, timed_out=False, signal=None, remaining_pids=[])

    def execute(self, locations, fake=None):
        out = locations['results'] / 'run'
        with patch.object(f5, '_locations', return_value=locations), \
             patch.object(f5, '_git_state', return_value=dict(branch='fixture', working_tree_changed=True)), \
             patch.object(f5.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 1024**3)), \
             patch.object(f5, 'run_process', side_effect=fake or self.fake_process) as launch:
            result = f5.execute(SimpleNamespace(out=out))
        return result, launch

    def test_two_interpreters_exact_paths_budgets_and_unique_eight_field_registry(self):
        with TemporaryDirectory() as temporary:
            locations = self.locations(Path(temporary))
            result, launch = self.execute(locations)
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(launch.call_count, 2)
            first, second = launch.call_args_list
            trace, parse = first.args[0], second.args[0]
            self.assertEqual(trace[:7], [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '90', '--python'])
            self.assertEqual(trace[7], str(locations['intake_python']))
            self.assertEqual(parse[0], str(locations['orbit_python']))
            self.assertEqual(parse[parse.index('--mlir') + 1], result['phase_results']['trace']['mlir']['path'])
            self.assertEqual(parse[parse.index('--profile') + 1], str(locations['profile']))
            self.assertEqual((first.kwargs['timeout'], second.kwargs['timeout']), (120, 30))
            for call in (first, second):
                self.assertEqual((call.kwargs['mem_gib'], call.kwargs['fsize_gib']), (4, 8))
                self.assertEqual(call.kwargs['env']['OPENBLAS_NUM_THREADS'], '1')
            rows = [json.loads(line) for line in (locations['results'] / 'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})
            self.assertEqual((rows[0]['round'], rows[0]['mode']), ('preparation', f5.MODE))
            self.assertEqual(rows[0]['metrics'], result['metrics'])
            self.assertEqual(sum(span['exclusive_ns'] for span in result['timing']['spans']),
                             result['timing']['spans'][0]['inclusive_ns'])
            for key in ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency'):
                self.assertIsNone(result['metrics'][key])
            for key in ('llm_calls', 'model_runs', 'optimizer_runs', 'keygen_runs', 'real_ckks_attempts'):
                self.assertEqual(result['metrics'][key], 0)

    def test_trace_failure_blocks_parse_and_retains_phase_result(self):
        with TemporaryDirectory() as temporary:
            locations = self.locations(Path(temporary))
            def fail(argv, **kwargs):
                process = self.fake_process(argv, **kwargs)
                work = Path(argv[argv.index('--work-dir') + 1])
                (work / 'phase_result.json').write_text(json.dumps(dict(status='dependency_missing',
                                                                       error='fixed missing dependency', trace_runs=1)))
                process['exit_code'] = 1
                return process
            result, launch = self.execute(locations, fail)
            self.assertEqual(result['status'], 'dependency_missing')
            launch.assert_called_once()
            self.assertNotIn('parse', result['phase_results'])
            self.assertEqual(result['metrics']['parse_runs'], 0)
            self.assertTrue((locations['results'] / 'run/trace-worker/phase_result.json').is_file())
            self.assertTrue((locations['results'] / 'run/result.json').is_file())

    def test_timeout_and_residual_process_prevent_parse(self):
        for field, value, status in [('timed_out', True, 'timeout'),
                                     ('remaining_pids', [123], 'process_failed')]:
            with self.subTest(field=field), TemporaryDirectory() as temporary:
                locations = self.locations(Path(temporary))
                def fail(argv, **kwargs):
                    process = self.fake_process(argv, **kwargs)
                    process[field] = value
                    return process
                result, launch = self.execute(locations, fail)
                self.assertEqual(result['status'], status)
                launch.assert_called_once()
                self.assertEqual(result['processes'][0][field], value)

    def test_disk_budget_failure_preserves_result_and_registry_without_child(self):
        with TemporaryDirectory() as temporary:
            locations = self.locations(Path(temporary))
            with patch.object(f5, 'FREE_BYTES', 21 * 1024**3):
                result, launch = self.execute(locations)
            launch.assert_not_called()
            self.assertEqual(result['status'], 'dependency_missing')
            self.assertEqual(result['metrics']['trace_runs'], 0)
            self.assertEqual(len((locations['results'] / 'registry.jsonl').read_text().splitlines()), 1)

    def test_case_artifact_budget_stops_before_new_work(self):
        with patch.object(f5, '_case_size', return_value=f5.CASE_BYTES + 1):
            with self.assertRaises(f5.PhaseStopped) as caught:
                f5._check_budget(Path('/fixture/out'))
        self.assertEqual(caught.exception.status, 'artifact_budget_exceeded')

    def test_terminal_metadata_size_changes_result_metrics_and_registry_without_new_action(self):
        with TemporaryDirectory() as temporary:
            locations = self.locations(Path(temporary))
            measured_size = f5._case_size
            base = f5.CASE_BYTES - 16

            def metadata_fixture(out):
                # Simulate prior artifacts close to the unchanged 4 GiB allowance.
                return base if (Path(out) / 'stages.jsonl').exists() else measured_size(out)

            with patch.object(f5, '_case_size', side_effect=metadata_fixture):
                result, launch = self.execute(locations)
            launch.assert_called()
            self.assertEqual(launch.call_count, 2)
            self.assertEqual([call.args[0][call.args[0].index('--phase') + 1]
                              for call in launch.call_args_list], ['trace', 'parse'])
            path = locations['results'] / 'run/result.json'
            saved = json.loads(path.read_text())
            registry = [json.loads(line) for line in
                        (locations['results'] / 'registry.jsonl').read_text().splitlines()]
            self.assertEqual(saved, result)
            self.assertEqual(result['status_before_artifact_budget'], 'ok')
            self.assertEqual(result['status'], 'artifact_budget_exceeded')
            self.assertEqual(result['metrics']['status'], result['status'])
            self.assertFalse(result['metrics']['case_artifact_budget_passed'])
            self.assertEqual(result['case_artifact_bytes'], base + path.stat().st_size)
            self.assertGreater(result['case_artifact_bytes'], f5.CASE_BYTES)
            self.assertEqual(result['metrics']['case_artifact_bytes'], result['case_artifact_bytes'])
            self.assertEqual(len(registry), 1)
            self.assertEqual(registry[0]['metrics'], result['metrics'])

    def test_existing_directory_and_evidence_are_preserved(self):
        with TemporaryDirectory() as temporary:
            locations = self.locations(Path(temporary))
            self.execute(locations)
            before = (locations['results'] / 'registry.jsonl').read_text()
            with self.assertRaises(ValueError):
                self.execute(locations)
            self.assertEqual((locations['results'] / 'registry.jsonl').read_text(), before)

    def test_deadline_constants_and_alarm_raise_finite_budget_error(self):
        self.assertEqual((f5.TRACE_SECONDS, f5.NIX_SECONDS, f5.TRACE_CHILD_SECONDS,
                          f5.PARSE_SECONDS, f5.OVERALL_SECONDS), (60, 90, 120, 30, 180))
        with patch.object(f5.signal, 'getsignal', return_value='previous'), \
             patch.object(f5.signal, 'getitimer', return_value=(0, 0)), \
             patch.object(f5.signal, 'signal') as handler, \
             patch.object(f5.signal, 'setitimer') as timer:
            with self.assertRaises(f5.PhaseStopped) as caught:
                with f5._wall_limit(60):
                    handler.call_args_list[0].args[1](None, None)
            self.assertEqual(caught.exception.status, 'timeout')
            self.assertEqual([call.args[1] for call in timer.call_args_list], [60, 0])

    def test_phase_cli_requires_explicit_paths_before_any_work(self):
        for argv in ([], ['--phase', 'trace', '--work-dir', '/fixture/work'],
                     ['--phase', 'parse', '--work-dir', '/fixture/work', '--orbit-root', '/fixture/orbit']):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), \
                 patch.object(f5, 'execute') as outer, patch.object(f5, 'run_phase') as inner:
                with self.assertRaises(SystemExit) as caught:
                    f5.main(argv)
                self.assertEqual(caught.exception.code, 2)
                outer.assert_not_called()
                inner.assert_not_called()
