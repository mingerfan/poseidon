"""R22 tiny protocol fixtures; no literature model, checkpoint or real trace."""

from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def benchmark_source(count=19):
    return ("import r22_forbidden_top_level_import\n"
            "raise AssertionError('top-level code must not execute')\n"
            "def getModel():\n"
            "    raise AssertionError('official weight loader must not execute')\n"
            "@hc.func('c')\n"
            "def ResNetSiLU64k(ctxt):\n"
            "    original = getModel()\n"
            "    for index in range(" + str(count) + "):\n"
            "        ctxt = HE_SiLU('act_' + str(index), ctxt)\n"
            "    return ctxt\n"
            "hc.save(dirs='forbidden', cst_dirs='forbidden')\n")


def graph_fixture():
    return dict(inputs=['cipher_input'], outputs=['actual_output'],
                nodes=[dict(id='cipher_input', op='input', level=None, scale=None,
                            weight=1, op_descr={}),
                       dict(id='actual_rotate', op='rotate', level=None, scale=None,
                            weight=1, op_descr=dict(offset=7)),
                       dict(id='actual_output', op='add', level=None, scale=None,
                            weight=2, op_descr={})],
                edges=[dict(source='cipher_input', target='actual_rotate', weight=3),
                       dict(source='actual_rotate', target='actual_output', weight=5)])


class FakeGraph:
    def __init__(self, record):
        self.record = record
        self.inputs = set(record['inputs'])
        self.outputs = set(record['outputs'])

    def nodes(self, data=False):
        return [(row['id'], {key: value for key, value in row.items() if key != 'id'})
                for row in self.record['nodes']]

    def edges(self, data=False):
        return [(row['source'], row['target'],
                 {key: value for key, value in row.items() if key not in ('source', 'target')})
                for row in self.record['edges']]


class OrbitModelCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        from pt_agent.experiments import r22_orbit_model_worker as worker
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        cls.worker, cls.Recorder = worker, StageRecorder

    def setUp(self):
        temporary = TemporaryDirectory(prefix='r22-inner-')
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.frontend = self.work / 'frontend'
        self.frontend.mkdir()
        self.deps = self.work / 'python-deps'
        (self.deps / 'einops').mkdir(parents=True)
        (self.deps / 'einops/__init__.py').write_text("__version__ = '0.8.1'\n")
        self.source = self.work / 'ResNetSiLU64k.py'
        self.source.write_text(benchmark_source())
        self.profile = self.work / 'orbit' / self.worker.PROFILE
        self.profile.parent.mkdir(parents=True)
        self.profile.write_text(json.dumps(dict(polynomialDegree=131072)))
        self.args = SimpleNamespace(phase='trace', work_dir=self.work,
            frontend=self.frontend, benchmark=self.source, deps_dir=self.deps,
            model_source=self.work / 'original-model.py', checkpoint=self.work / 'weights.pt',
            poly_dir=self.work / 'local-poly/poly', orbit_root=self.work / 'orbit',
            profile=self.profile, mlir=self.work / 'trace/model.mlir',
            r13_result=self.work / 'r13-result.json', r16_result=self.work / 'r16-result.json')
        self.recorder = self.Recorder(self.work / 'events.jsonl')

    def _namespace(self):
        registered = []
        hc = SimpleNamespace()

        def decorate(signature):
            def apply(body):
                registered.append((signature, body))
                return body
            return apply

        hc.func = Mock(side_effect=decorate)
        return dict(hc=hc, getModel=Mock(return_value=object()),
                    HE_SiLU=Mock(side_effect=lambda name, value: value)), registered

    def _trace(self, *, count=19, save_error=None, load_error=None, mutate=True):
        self.source.write_text(benchmark_source(count))
        value, model, registered, delegated = object(), object(), [], []
        hc = SimpleNamespace(ctxt=object(), lt=SimpleNamespace(finalize=Mock()))

        def register(signature):
            def decorate(body):
                registered.append((signature, body))
                return body
            return decorate

        def save(*, dirs, cst_dirs):
            self.assertEqual(len(registered), 1)
            self.assertEqual(registered[0][0], 'c')
            self.assertIs(registered[0][1](value), value)
            if save_error is not None:
                raise save_error
            path = Path(dirs) / 'fixed-model.mlir'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixed protocol fixture, never compiled\n')
            Path(cst_dirs).mkdir(parents=True, exist_ok=True)
            (Path(cst_dirs) / 'fixed.cst').write_text('fixture only\n')
            if mutate:
                os.chdir(self.frontend)
            return str(path)

        def load_expr(frontend):
            self.assertEqual(os.environ['HECATE'], str(frontend))
            os.environ['PATH'] = '/r22-fixture-injected:' + os.environ.get('PATH', '')
            sys.modules['orbit_expr_check'] = hc
            return hc

        def delegate(name, item):
            delegated.append((name, item))
            self.assertIs(item, value)
            return item

        mpcb = SimpleNamespace(**{name: Mock() for name in
            ('CascadeConv', 'CascadeDS', 'CascadePool', 'shapeClosure')})
        func = SimpleNamespace(**{name: Mock() for name in
            ('HE_ConvBN', 'HE_DS', 'HE_Pool', 'HE_Linear')})
        func.HE_SiLU = Mock(side_effect=delegate)
        poly = SimpleNamespace()

        def load_poly(poly_dir, module):
            self.assertIs(module, hc)
            for name, item in (('poly', ModuleType('poly')), ('poly.MPCB', mpcb),
                               ('poly.Poly', poly), ('poly.Func', func)):
                sys.modules[name] = item
            sys.path.insert(0, str(self.work / 'injected-path'))
            return mpcb, poly, func

        hc.func, hc.save = Mock(side_effect=register), Mock(side_effect=save)
        self.last_hc = hc
        result = self.worker._initial_result('trace')
        with ExitStack() as stack:
            hooks = dict(_load_expr=load_expr, _load_poly=load_poly,
                         _load_einops=lambda path: SimpleNamespace(
                             __file__=str(self.deps / 'einops/__init__.py'), __version__='0.8.1'))
            for name, target in hooks.items():
                stack.enter_context(patch.object(self.worker, name, side_effect=target))
            loader = stack.enter_context(patch.object(self.worker, '_load_model',
                return_value=model, side_effect=load_error))
            self.worker.trace_model(self.args, self.recorder, result)
        return result, hc, loader, delegated, model

    def test_ast_selects_only_original_function_and_preserves_body(self):
        namespace, registered = self._namespace()
        body = self.worker.load_benchmark_function(self.source, namespace)
        token = object()
        self.assertIs(body(token), token)
        self.assertEqual(len(registered), 1)
        namespace['hc'].func.assert_called_once_with('c')
        namespace['getModel'].assert_called_once_with()
        self.assertEqual(namespace['HE_SiLU'].call_count, 19)

    def test_ast_rejects_missing_duplicate_async_and_different_name(self):
        cases = ('def unrelated(x):\n    return x\n',
                 benchmark_source() + "\ndef ResNetSiLU64k(x):\n    return x\n",
                 "@hc.func('c')\nasync def ResNetSiLU64k(x):\n    return x\n",
                 benchmark_source().replace('def ResNetSiLU64k(', 'def ResNetSiLU16k('))
        for source in cases:
            with self.subTest(source=source):
                self.source.write_text(source)
                namespace, _ = self._namespace()
                with self.assertRaises(ValueError):
                    self.worker.load_benchmark_function(self.source, namespace)

    def test_ast_rejects_changed_signature_or_extra_decorator(self):
        for decorator in ("@hc.func('c,c')", '@hc.func', "@other.func('c')",
                          "@extra\n@hc.func('c')"):
            with self.subTest(decorator=decorator):
                self.source.write_text(benchmark_source().replace("@hc.func('c')", decorator))
                namespace, _ = self._namespace()
                with self.assertRaises(ValueError):
                    self.worker.load_benchmark_function(self.source, namespace)

    def test_trace_delegates_nineteen_activations_and_finalizes_once(self):
        previous_cwd = Path.cwd()
        self.addCleanup(os.chdir, previous_cwd)
        result, hc, loader, delegated, _ = self._trace()
        loader.assert_called_once()
        self.assertEqual([name for name, _ in delegated], ['act_' + str(i) for i in range(19)])
        self.assertEqual(result['get_model_calls'], 1)
        self.assertEqual(result['activation_calls'], 19)
        self.assertTrue(result['orbit_model_trace_passed'])
        self.assertEqual(result['save_attempts'], 1)
        self.assertEqual(result['save_completed'], 1)
        self.assertEqual(result['finalize_attempts'], 1)
        self.assertEqual(result['finalize_completed'], 1)
        hc.save.assert_called_once()
        hc.lt.finalize.assert_called_once_with(hc.ctxt)

    def test_trace_restores_hecate_path_modules_sys_path_and_cwd(self):
        keys = ('hecate', 'poly', 'poly.MPCB', 'poly.Poly', 'poly.Func', 'orbit_expr_check')
        prior_modules = {key: ModuleType('r22_original_' + key) for key in keys}
        with patch.dict(sys.modules, prior_modules), patch.dict(os.environ,
                HECATE='r22-original-hecate', PATH='r22-original-path'):
            previous_path, previous_cwd = list(sys.path), Path.cwd()
            self.addCleanup(os.chdir, previous_cwd)
            self._trace()
            self.assertEqual(os.environ['HECATE'], 'r22-original-hecate')
            self.assertEqual(os.environ['PATH'], 'r22-original-path')
            self.assertEqual(sys.path, previous_path)
            self.assertEqual(Path.cwd(), previous_cwd)
            for key in keys:
                self.assertIs(sys.modules[key], prior_modules[key])

    def test_trace_restores_previously_absent_hecate_and_path(self):
        with patch.dict(os.environ):
            os.environ.pop('HECATE', None)
            os.environ.pop('PATH', None)
            previous_cwd = Path.cwd()
            self.addCleanup(os.chdir, previous_cwd)
            self._trace()
            self.assertNotIn('HECATE', os.environ)
            self.assertNotIn('PATH', os.environ)

    def test_save_failure_still_finalizes_and_restores_state(self):
        previous_cwd = Path.cwd()
        self.addCleanup(os.chdir, previous_cwd)
        captured = []
        original = self.worker._initial_result

        def initial(phase):
            result = original(phase)
            captured.append(result)
            return result

        with patch.object(self.worker, '_initial_result', side_effect=initial), \
                patch.dict(os.environ, HECATE='before-failure', PATH='original-path'):
            with self.assertRaisesRegex(RuntimeError, 'fixed save failure'):
                self._trace(save_error=RuntimeError('fixed save failure'))
            self.assertEqual(os.environ['HECATE'], 'before-failure')
            self.assertEqual(os.environ['PATH'], 'original-path')
            self.assertEqual(captured[0]['save_attempts'], 1)
            self.assertEqual(captured[0]['save_completed'], 0)
            self.assertEqual(captured[0]['finalize_attempts'], 1)
            self.assertEqual(captured[0]['finalize_completed'], 1)
            self.last_hc.lt.finalize.assert_called_once_with(self.last_hc.ctxt)

    def test_wrong_activation_count_stops_without_success(self):
        for count in (18, 20):
            self.args.work_dir = self.work / ('wrong-count-' + str(count))
            self.args.work_dir.mkdir()
            with self.subTest(count=count), self.assertRaises(self.worker.WorkerFailure):
                self._trace(count=count, mutate=False)

    def test_profile_alias_uses_original_64k_metadata_and_keeps_other_fields(self):
        params = SimpleNamespace(poly_deg=32768, max_slot=16384, mode='compile',
                                 sysname='Orbit', threads=1, unchanged_marker=object())
        marker = params.unchanged_marker
        constructor = Mock(return_value=params)
        actual, metadata = self.worker._profile_params(self.profile, constructor)
        self.assertIs(actual, params)
        constructor.assert_called_once_with(str(self.profile), 'Orbit', 'compile', threads=1)
        self.assertEqual((actual.poly_deg, actual.max_slot), (131072, 65536))
        self.assertIs(actual.unchanged_marker, marker)
        self.assertIn('polynomialDegree', json.dumps(metadata))
        self.assertIn('32768', json.dumps(metadata))

    def test_profile_rejects_missing_wrong_and_nonintegral_degree(self):
        for record in ({}, dict(polynomialDegree=32768), dict(polynomialDegree=131072.5),
                       dict(polynomialDegree=True)):
            with self.subTest(record=record):
                self.profile.write_text(json.dumps(record))
                constructor = Mock(return_value=SimpleNamespace(poly_deg=32768, max_slot=16384))
                with self.assertRaises(ValueError):
                    self.worker._profile_params(self.profile, constructor)

    def test_graph_accepts_general_dag_and_preserves_raw_metadata(self):
        record = graph_fixture()
        before = copy.deepcopy(record)
        checked = self.worker._check_graph(record)
        self.assertTrue(checked['passed'])
        self.assertEqual(record, before)
        self.assertEqual(record['nodes'][1]['op_descr']['offset'], 7)
        self.assertIsNone(record['nodes'][1]['level'])

    def test_graph_rejects_unknown_nodes_cycles_unreachable_and_multiple_io(self):
        variants = []
        row = graph_fixture(); row['outputs'] = ['missing']; variants.append(row)
        row = graph_fixture(); row['edges'].append(dict(source='actual_output', target='cipher_input', weight=1)); variants.append(row)
        row = graph_fixture(); row['edges'].pop(); variants.append(row)
        row = graph_fixture(); row['inputs'].append('actual_rotate'); variants.append(row)
        row = graph_fixture(); row['outputs'].append('actual_rotate'); variants.append(row)
        for record in variants:
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.worker._check_graph(record)

    def test_parse_uses_only_original_constructor_and_builder_once(self):
        (self.work / 'trace').mkdir()
        self.args.mlir.write_text('fixed parser fixture; never real MLIR\n')
        params = SimpleNamespace(poly_deg=32768, max_slot=16384, mode='compile',
                                 sysname='Orbit', threads=1)
        constructor = Mock(return_value=params)
        builder = Mock(return_value=FakeGraph(graph_fixture()))
        result = self.worker._initial_result('parse')
        with patch.object(self.worker, '_load_parser', return_value=(constructor, builder)):
            self.worker.parse_model(self.args, self.recorder, result)
        constructor.assert_called_once_with(str(self.profile), 'Orbit', 'compile', threads=1)
        builder.assert_called_once_with(str(self.args.mlir), params)
        self.assertEqual(result['params_calls'], 1)
        self.assertEqual(result['build_from_mlir_calls'], 1)
        self.assertTrue(result['tdag_parse_passed'])
        self.assertEqual(json.loads((self.work / 'tdag.json').read_text()), graph_fixture())

    def test_parse_retains_invalid_raw_graph_before_rejecting(self):
        (self.work / 'trace').mkdir()
        self.args.mlir.write_text('fixed parser fixture; never real MLIR\n')
        record = graph_fixture()
        record['edges'].append(dict(source='actual_output', target='cipher_input', weight=1))
        params = SimpleNamespace(poly_deg=32768, max_slot=16384, mode='compile',
                                 sysname='Orbit', threads=1)
        result = self.worker._initial_result('parse')
        with patch.object(self.worker, '_load_parser', return_value=(Mock(return_value=params),
                Mock(return_value=FakeGraph(record)))), self.assertRaises(ValueError):
            self.worker.parse_model(self.args, self.recorder, result)
        self.assertEqual(json.loads((self.work / 'tdag.json').read_text()), record)
        self.assertFalse(result['tdag_parse_passed'])


if __name__ == '__main__':
    unittest.main()
