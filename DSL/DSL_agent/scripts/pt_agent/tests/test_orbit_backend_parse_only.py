"""Synthetic source/cache metadata and fake Go processes; no Go tool execution."""
from argparse import Namespace
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pt_agent.experiments import orbit_backend_parse_only as entry


class OrbitParserChecks(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    @staticmethod
    def clean(**changes):
        result = dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])
        result.update(changes);return result

    def fixture(self):
        orbit = self.root/'orbit';module = orbit/'backend/lattigo';package = module/'lowering/fhe';package.mkdir(parents=True)
        for name in ('main.go', 'parser.go', 'scheme.go', 'exeorder.go'):(package/name).write_text('package main\n')
        sdk = self.root/'sdk';(sdk/'bin').mkdir(parents=True);(sdk/'src').mkdir()
        (sdk/'bin/go').write_text('existing fixture tool');(sdk/'bin/go').chmod(0o700)
        (sdk/'src/fixture.go').write_text('package fixture\n')
        cache = self.root/'cache';cache.mkdir();(cache/'fixture.go').write_text('package dependency\n')
        build = self.root/'shared-build';(build/'work').mkdir(parents=True)
        shutil.copytree(cache, build/'work/module-cache')
        (build/'orbit-fhe').write_bytes(b'existing shared interpreter metadata fixture')
        source_case = self.root/'optimizer';source_case.mkdir()
        mlir = source_case/'optimized.mlir';mlir.write_text('tiny synthetic Earth source fixture\n')
        graph = dict(passed=True, acyclic=True, node_count=4, edge_count=3,
            op_counts=dict(input=1, constant=1, add=1, bootstrap=1), inputs=['arg0'], outputs=['2'], placement_bootstrap_nodes=1)
        (source_case/'result.json').write_text(json.dumps(dict(phase_results=dict(validate=dict(graph_check=graph)))))
        source = dict(source_result=str(source_case/'result.json'), model=entry.shared.MODEL,
            optimized_mlir=dict(path=str(mlir), bytes=mlir.stat().st_size),
            optimizer_params=dict(max_slot=65536, bts_ub=16), placement_bootstrap_nodes=1)
        (build/'result.json').write_text(json.dumps(dict(status='ok', runtime_interpreter_compiled=True,
            go_build_attempts=1, go_build_completed=1, source=source, process=self.clean(),
            package=dict(interpreter_package=str(package)), binary=dict(path=str(build/'orbit-fhe'), bytes=(build/'orbit-fhe').stat().st_size))))
        audit = self.root/'audit.json';files = []
        for scope, base, relative in (('module', module, 'lowering/fhe/main.go'), ('sdk', sdk, 'src/fixture.go'), ('cache', cache, 'fixture.go')):
            path = base/relative
            files.append(dict(scope=scope, relative=relative, bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
        audit.write_text(json.dumps(dict(status='passed_source_no_model_context_keys_or_payload_initializers',
            unresolved_imports=[], roots=dict(module=str(module), sdk=str(sdk), cache=str(cache)), audited_files=files,
            explicit_review=dict(model_context_initializers=0, keygen_initializers=0, model_arithmetic_initializers=0, external_weight_or_constant_payload_reads=0))))
        args = Namespace(out=self.root/'case', source_case=source_case, shared_build_case=build,
            orbit_root=orbit, go_sdk=sdk, initialization_audit=audit, registry=self.root/'registry.jsonl', run=True)
        expected = {key: graph[key] for key in ('node_count', 'edge_count', 'op_counts', 'inputs', 'outputs', 'placement_bootstrap_nodes')}
        expected['output_id'] = 2
        checked = dict(optimized_source=source, expected_graph=expected, module=str(module), module_cache=str(build/'work/module-cache'))
        return args, source, checked

    @staticmethod
    def parsed(checked):
        source = checked['optimized_source']
        result = dict(status='ok', mlir_path=source['optimized_mlir']['path'], mlir_bytes=source['optimized_mlir']['bytes'],
            mlir_text_reads=2, parsed_nodes=4, parsed_operations=3, input_count=1, output_id=2, edge_count=3,
            operand_edges_with_duplicates=3, op_counts=dict(input=1, constant=1, add=1, bootstrap=1), static_bootstrap_nodes=1,
            topological_order_count=3, nodes=[dict(id=i) for i in range(3)], official_ReadFile_calls=1,
            official_parseMLIROperation_calls=3, official_getOptimalRunOrder_calls=1, official_getMaxWorkingPoolSize_calls=1)
        result.update({key: 0 for key in entry.ZERO_FIELDS[:9]});return result

    def test_source_admission_accepts_exact_shared_build_and_rebased_copied_cache(self):
        args, source, checked = self.fixture()
        with patch.object(entry.shared, 'source_gate', return_value=source):result = entry.source_gate(args)
        self.assertEqual(result['expected_graph'], checked['expected_graph'])
        self.assertTrue(result['initialization']['cache_source_rebased'])
        self.assertTrue(result['initialization']['shared_public_scalar_initializers'])

    def test_shared_binary_or_model_source_drift_is_rejected(self):
        args, source, checked = self.fixture();path = args.shared_build_case/'result.json'
        good = entry.read(path)
        with patch.object(entry.shared, 'source_gate', return_value=source):
            for name in ('source', 'binary', 'process'):
                changed = copy.deepcopy(good)
                if name == 'source':changed['source']['optimized_mlir']['path'] = '/foreign/model.mlir'
                elif name == 'binary':changed['binary']['bytes'] += 1
                else:changed['process']['remaining_pids'] = [42]
                path.write_text(json.dumps(changed))
                with self.subTest(name=name), self.assertRaises(ValueError):entry.source_gate(args)

    def test_initialization_gate_rejects_unknown_imports_unsafe_initializers_and_metadata_drift(self):
        args, source, checked = self.fixture();original = entry.read(args.initialization_audit)
        for name in ('unknown', 'keygen', 'payload', 'changed'):
            record = copy.deepcopy(original)
            if name == 'unknown':record['unresolved_imports'] = ['unknown']
            elif name == 'changed':record['audited_files'][0]['mtime_ns'] += 1
            else:record['explicit_review']['keygen_initializers' if name == 'keygen' else 'external_weight_or_constant_payload_reads'] = 1
            args.initialization_audit.write_text(json.dumps(record))
            with self.subTest(name=name), self.assertRaises(ValueError):
                entry.initialization_gate(args.initialization_audit, Path(checked['module']), args.go_sdk, Path(checked['module_cache']))

    def test_initialization_gate_rejects_path_escape_and_sdk_identity_change(self):
        args, source, checked = self.fixture();original = entry.read(args.initialization_audit)
        for change in ('path', 'sdk'):
            record = copy.deepcopy(original)
            if change == 'path':record['audited_files'][0]['relative'] = '../outside.go'
            else:record['roots']['sdk'] = str(self.root/'foreign')
            args.initialization_audit.write_text(json.dumps(record))
            with self.subTest(change=change), self.assertRaises(ValueError):
                entry.initialization_gate(args.initialization_audit, Path(checked['module']), args.go_sdk, Path(checked['module_cache']))

    def test_prepare_uses_main_overlay_and_copies_cache_without_changing_original_sources(self):
        args, source, checked = self.fixture();args.out.mkdir()
        original = (Path(checked['module'])/'lowering/fhe/main.go').read_text()
        work = entry.prepare(args, args.out, checked)
        overlay = entry.read(args.out/'overlay.json')['Replace']
        self.assertEqual(overlay, {str(Path(checked['module'])/'lowering/fhe/main.go'): str(args.out/'source/parser-main.go')})
        self.assertEqual((Path(checked['module'])/'lowering/fhe/main.go').read_text(), original)
        self.assertEqual((work/'telemetry/mode').read_text(), 'off\n')
        self.assertTrue((work/'module-cache/fixture.go').is_file())
        self.assertFalse((args.out/'constants').exists())

    def test_commands_use_official_package_and_original_mlir_metadata_only(self):
        args, source, checked = self.fixture()
        build = entry.build_command(args.go_sdk, args.out)
        self.assertIn('-overlay', build);self.assertEqual(build[-1], './lowering/fhe')
        command = entry.parser_command(args.out, checked)
        self.assertEqual(command[command.index('--mlir')+1], source['optimized_mlir']['path'])
        self.assertEqual(command[command.index('--slots')+1], '65536')
        for forbidden in ('--input', '--cons', '--heMode', '--plain'):self.assertNotIn(forbidden, command)
        self.assertIn('--self-test', entry.parser_command(args.out, checked, selftest=True))

    def test_client_source_reuses_original_parser_and_scheduler_without_crypto_or_payload_calls(self):
        for call in ('holder.ReadFile(path)', 'holder.parseMLIROperation(line)', 'holder.getOptimalRunOrder', 'holder.getMaxWorkingPoolSize'):
            self.assertIn(call, entry.CLIENT)
        for call in ('.Run(', '.RunBatch(', '.preprocess(', '.createContext(', '.loadConstants(', '.processInputs(', '.encode(', '.decode(', 'NewParametersFromLiteral(', 'NewKeyGenerator('):
            self.assertNotIn(call, entry.CLIENT)
        self.assertIn('func selftest()', entry.CLIENT)
        self.assertIn('if len(order)!=len(operations)', entry.CLIENT)
        self.assertIn('edgeCount+=len(children)', entry.CLIENT)
        self.assertIn('operandEdges+=len(term.Children)', entry.CLIENT)

    def test_parser_result_requires_full_counts_output_identity_order_and_zero_crypto(self):
        args, source, checked = self.fixture();good = self.parsed(checked)
        self.assertTrue(entry.check_parsed(good, checked)['passed'])
        for name in ('nodes', 'output', 'boot', 'key', 'source', 'parser', 'edges'):
            record = copy.deepcopy(good)
            if name == 'nodes':record['nodes'][1]['id'] = 0
            elif name == 'output':record['output_id'] = 1
            elif name == 'boot':record['static_bootstrap_nodes'] = 2
            elif name == 'key':record['keygen_runs'] = 1
            elif name == 'source':record['mlir_path'] = '/foreign/model.mlir'
            elif name == 'parser':record['official_parseMLIROperation_calls'] = 2
            else:record['edge_count'] = 2
            with self.subTest(name=name), self.assertRaises(ValueError):entry.check_parsed(record, checked)

    def outer(self, failure=None, run=True):
        args, source, checked = self.fixture();args.run = run;calls = []
        def process(command, **options):
            calls.append((command, options));Path(options['log']).write_text('fake subprocess\n');Path(options['resources']).write_text('')
            if command[1] == 'build':
                (args.out/'orbit-parser').write_bytes(b'fake compiled parser metadata')
                return self.clean(exit_code=1 if failure == 'build' else 0)
            target = Path(command[command.index('--out')+1])
            if '--self-test' in command:
                record = dict(status='ok', selftests=6, model_parses=0, **{key: 0 for key in entry.ZERO_FIELDS[:9]})
                if failure == 'selftest':record['selftests'] = 5
            else:
                record = self.parsed(checked)
                if failure == 'parse':record['static_bootstrap_nodes'] = 0
            target.write_text(json.dumps(record)+'\n')
            return self.clean(timed_out=failure == 'timeout' and '--mlir' in command)
        with ExitStack() as stack:
            stack.enter_context(patch.object(entry.paths, 'RESULTS_DIR', self.root))
            stack.enter_context(patch.object(entry.shared, 'source_gate', return_value=source))
            stack.enter_context(patch.object(entry, '_available_ram_bytes', return_value=16*entry.GIB))
            stack.enter_context(patch.object(entry.shutil, 'disk_usage', return_value=NS(free=100*entry.GIB)))
            stack.enter_context(patch.object(entry, 'run_process', side_effect=process))
            result = entry.execute(args)
        return args, result, calls

    def test_preparation_never_runs_go_and_keeps_consumption_unknown(self):
        args, result, calls = self.outer(run=False)
        self.assertEqual(calls, []);self.assertEqual(result['status'], 'prepared')
        self.assertIsNone(result['backend_parser_consumption_verified']);self.assertTrue(result['compiler_graph_completed'])
        self.assertIsNone(result['model_specific_native_codegen'])

    def test_fake_complete_pipeline_preserves_resource_limits_registry_and_completion(self):
        args, result, calls = self.outer()
        self.assertEqual(result['status'], 'ok');self.assertTrue(result['backend_parser_consumption_verified'])
        self.assertEqual(result['completed'], dict(build=1, selftest=1, parse=1));self.assertEqual(len(calls), 3)
        self.assertEqual([row[1]['mem_gib'] for row in calls], [12, 12, 12])
        self.assertEqual([row[1]['fsize_gib'] for row in calls], [8, 8, 8])
        for (command, options), seconds in zip(calls, (600, 120, 300)):self.assertLessEqual(options['timeout'], seconds)
        records=[json.loads(line) for line in args.registry.read_text().splitlines()]
        self.assertEqual(len(records), 1);self.assertEqual(set(records[0]), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})
        completion = entry.read(args.out/'completion.json')
        self.assertTrue(completion['registry_persisted']);self.assertTrue(completion['exclusive_attribution_ready'])
        self.assertEqual(completion['case_artifact_bytes'], entry.artifact_bytes(args.out))

    def test_build_failure_stops_before_any_parser_launch(self):
        args, result, calls = self.outer(failure='build')
        self.assertEqual(result['status'], 'error');self.assertEqual(len(calls), 1)
        self.assertEqual(result['attempted'], dict(build=1, selftest=0, parse=0))

    def test_failed_public_selftest_stops_before_model_parser(self):
        args, result, calls = self.outer(failure='selftest')
        self.assertEqual(len(calls), 2);self.assertEqual(result['attempted']['parse'], 0)
        self.assertIsNone(result['backend_parser_consumption_verified'])

    def test_static_graph_mismatch_retains_model_result_without_claiming_consumption(self):
        args, result, calls = self.outer(failure='parse')
        self.assertEqual(len(calls), 3);self.assertEqual(result['status'], 'error')
        self.assertEqual(result['parsed']['static_bootstrap_nodes'], 0)
        self.assertEqual(result['completed']['parse'], 0);self.assertIsNone(result['backend_parser_consumption_verified'])

    def test_parser_timeout_is_distinct_and_original_observation_is_saved(self):
        args, result, calls = self.outer(failure='timeout')
        self.assertEqual(result['status'], 'timeout');self.assertTrue(result['wall_timeout_observed'])
        self.assertTrue(result['phases']['parse']['observation']['timed_out'])

    def test_terminal_budget_reserves_completion_and_registry_rejects_duplicate_identity(self):
        out = self.root/'terminal';out.mkdir();record = dict(status='ok', error=None)
        (out/'payload').write_bytes(b'x'*200)
        with patch.object(entry, 'CASE_BYTES', 1000), patch.object(entry, 'RESERVE_BYTES', 900):entry.terminal(out, record)
        self.assertEqual(record['status'], 'resource_stopped');self.assertEqual(record['artifact_bytes'], entry.artifact_bytes(out))
        for key in ('attempted', 'completed'):record[key] = {}
        for key in ('backend_parser_consumption_verified', 'compiler_graph_completed', 'model_specific_native_codegen', 'formal_latency', 'accuracy_acceptance', 'security_acceptance'):record[key] = None
        entry.append_registry(out, record, self.root/'registry.jsonl')
        with self.assertRaises(ValueError):entry.append_registry(out, record, self.root/'registry.jsonl')


if __name__ == '__main__':unittest.main()
