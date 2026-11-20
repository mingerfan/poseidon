"""R33 synthetic math and source contracts; no original graphs or data."""
import ast
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent import paths
from pt_agent.experiments import r33_orbit_optimized_math as math
from pt_agent.tests._artifacts import require

NUMPY_PYTHON = paths.ORBIT_ROOT.parent/'orbit-venv/bin/python'
NUMPY_ARTIFACT = NUMPY_PYTHON.parent if NUMPY_PYTHON.is_symlink() else NUMPY_PYTHON


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def binary(path, values):
    with path.open('wb') as stream:
        stream.write(struct.pack('<q', len(values)))
        for row in values:
            stream.write(struct.pack('<q', len(row)))
            stream.write(struct.pack('<'+'d'*len(row), *row))


def graph(operations):
    nodes = [dict(id='arg0', op='input', op_descr={})]
    edges = []
    for index, (kind, inputs, attrs) in enumerate(operations):
        name = str(index)
        nodes.append(dict(id=name, op=kind, op_descr=attrs))
        edges.extend(dict(source=source, target=name) for source in inputs)
    return dict(inputs=['arg0'], outputs=[str(len(operations)-1)], nodes=nodes, edges=edges)


class SourceCases(unittest.TestCase):
    def test_graph_missing_dependencies_and_invalid_identities_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'graph.json'
            valid = graph([('rotate',['arg0'],dict(offset=1)), ('add',['0','0'],{})])
            save(path, valid)
            program = math.load_program(path)
            self.assertEqual(program['incoming']['1'], ['0','0'])
            self.assertEqual(program['metadata']['repeated_operand_edges'], 1)
            for changed in (dict(valid, edges=[dict(source='absent',target='0')]),
                            graph([('upscale',['arg0'],dict(upFactor=-1))]),
                            graph([('bootstrap',['arg0'],dict(targetLevel=True))])):
                save(path, changed)
                with self.assertRaises(ValueError):math.load_program(path)

    def test_source_identity_requires_original_r25_and_optimized_r23(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = SimpleNamespace(**{name:base/(name+'.json') for name in math.SOURCE_NAMES})
            args.r23_result = base/'r23/result.json'
            args.graph = base/'r23/work/validate-worker/tdag.json'
            for name in math.SOURCE_NAMES:
                save(getattr(args,name), {})
            original_graph = base/'r22-graph.json';save(original_graph,{})
            source_paths = {name:dict(path=str(getattr(args,name))) for name in
                ('constants','probes','probe_source','r13_result','r22_result','r24_result')}
            source_paths['graph'] = dict(path=str(original_graph))
            w25 = dict(source_checked=True,native_packed_equivalence_verified=True,
                packed_forward_attempts=3,packed_forward_completed=3,math_check_attempts=3,math_check_completed=3,
                program_metadata=dict(constant_ids=list(range(1547))),source_summary=dict(paths=source_paths))
            r25 = dict(round='R25',model=math.MODEL,mode=math.original.MODE,status='ok',
                native_packed_equivalence_verified=True,worker=w25)
            r23 = dict(round='R23',model=math.MODEL,mode='orbit-resnet20-native-optimizer-preflight',status='ok',
                native_optimizer_completed=True,placement_artifact_checked=True,placement_bootstrap_nodes=19,
                phase_results=dict(validate=dict(status='ok',graph_check=dict(passed=True),tdag=str(args.graph))),
                source_records=dict(r22_result=str(args.r22_result)))
            save(args.r25_result,r25);save(args.r23_result,r23)
            with mock.patch.object(math.original,'_check_sources',return_value=({}, {}, {}, {})) as prior:
                summary,_,_ = math.check_sources(args)
                self.assertEqual(prior.call_args.args[0].graph, original_graph)
                self.assertEqual(summary['paths']['graph']['path'],str(args.graph.resolve()))
                for changed in (dict(r23,status='error'),dict(r23,placement_bootstrap_nodes=18)):
                    save(args.r23_result,changed)
                    with self.assertRaises(ValueError):math.check_sources(args)
                save(args.r23_result,r23)
                changed=copy.deepcopy(r25);changed['worker']['program_metadata']['constant_ids'].pop()
                save(args.r25_result,changed)
                with self.assertRaises(ValueError):math.check_sources(args)
                save(args.r25_result,r25)
                with self.assertRaises(ValueError):math.check_sources(SimpleNamespace(**dict(vars(args),constants=base/'wrong.cst')))

    def test_worker_cli_requires_all_explicit_sources_and_no_numeric_import_at_top_level(self):
        tree=ast.parse(Path(math.__file__).read_text())
        top_imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('numpy' in ast.unparse(n) or 'torch' in ast.unparse(n) for n in top_imports))
        argv=['--phase','worker']
        for name in math.SOURCE_NAMES:argv+=['--'+name.replace('_','-'),'/synthetic/'+name]
        argv+=['--work-dir','/synthetic/work']
        with mock.patch.object(math,'worker',return_value=dict(status='ok')) as worker:
            self.assertEqual(math.main(argv),0)
        self.assertEqual(worker.call_args.args[0].graph,Path('/synthetic/graph'))
        self.assertNotIn('registry', vars(worker.call_args.args[0]))


def numeric_cases():
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    import numpy as np

    class MathCases(unittest.TestCase):
        def test_header_index_reads_no_payload_and_repeated_materialization_stays_readonly(self):
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'values.cst';binary(path,[[1.,2.,3.],list(range(12))])
                with mock.patch.object(np,'fromfile',side_effect=AssertionError('index reads no values')):
                    indexed=math.index_constants(path,8,2)
                self.assertEqual(indexed['payload_values_read'],0)
                with math.LazyConstants(indexed) as reader:
                    first=reader.get(0);second=reader.get(0)
                    np.testing.assert_array_equal(first,np.array([1.,2.,3.,1.,2.,3.,1.,2.]))
                    self.assertFalse(first.flags.writeable);self.assertFalse(np.shares_memory(first,second))
                    np.testing.assert_array_equal(reader.get(1),np.arange(8,dtype=np.float64))
                    self.assertEqual(reader.summary()['materialization_attempts'],3)
                    self.assertEqual(reader.summary()['persistent_numeric_cache_bytes'],0)

        def test_truncation_and_effective_nonfinite_reject_while_long_tail_is_unobserved(self):
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'values.cst';binary(path,[list(range(8))+[float('nan')]])
                indexed=math.index_constants(path,8,1)
                self.assertEqual(indexed['entries'][0]['skipped_values'],1)
                with math.LazyConstants(indexed) as reader:
                    self.assertTrue(np.isfinite(reader.get(0)).all())
                    self.assertIsNone(reader.summary()['skipped_tail_finite'])
                path.write_bytes(path.read_bytes()[:-1])
                with self.assertRaises(ValueError):math.index_constants(path,8,1)
                binary(path,[[1.,float('nan')]])
                with math.LazyConstants(math.index_constants(path,8,1)) as reader:
                    with self.assertRaisesRegex(ValueError,'nonfinite'):reader.get(0)

        def test_repeated_dependencies_identity_and_rotation_retain_original_input(self):
            operations=[('constant',[],dict(value=0)),('mul',['arg0','0'],{}),
                ('upscale',['1'],dict(upFactor=11)),('rescale',['2'],{}),
                ('modswitch',['3'],dict(downFactor=2)),('bootstrap',['4'],dict(targetLevel=16)),
                ('rotate',['5'],dict(offset=1)),('add',['6','6'],{}),('add',['7','arg0'],{})]
            with tempfile.TemporaryDirectory() as directory:
                base=Path(directory);save(base/'graph.json',graph(operations));binary(base/'cst',[[2.]])
                program=math.load_program(base/'graph.json');result=math.initial_result();packed=np.arange(8,dtype=np.float64)
                with math.LazyConstants(math.index_constants(base/'cst',8,1)) as reader:
                    output=math.execute_program(program,reader,packed,result)
                    self.assertEqual(reader.reads,1)
                np.testing.assert_array_equal(output,4*np.roll(packed,-1)+packed)
                np.testing.assert_array_equal(packed,np.arange(8,dtype=np.float64))
                self.assertEqual(result['op_completed'],9)
                self.assertEqual(result['ideal_identity_nodes'],dict(rescale=1,upscale=1,modswitch=1,bootstrap=1))
                self.assertEqual(result['native_bootstrap_executions'],0)

        def test_nonfinite_numeric_operation_and_original_strict_math_keep_policy(self):
            with tempfile.TemporaryDirectory() as directory:
                base=Path(directory);save(base/'graph.json',graph([('mul',['arg0','arg0'],{})]));binary(base/'cst',[[1.]])
                with math.LazyConstants(math.index_constants(base/'cst',8,1)) as reader:
                    with self.assertRaises(FloatingPointError):
                        math.execute_program(math.load_program(base/'graph.json'),reader,np.full(8,1e308),math.initial_result())
            reference=np.ones((1,10),dtype=np.float64)
            self.assertTrue(math.original._check_math(reference,reference.copy())['passed'])
            self.assertFalse(math.original._check_math(reference,reference+1e-7)['passed'])
            self.assertEqual(math.original._check_math(reference,reference)['criterion'],'1e-12+1e-12*abs(reference)')

        def _run_fixture(self, directory, *, rejected=False):
            base=Path(directory);save(base/'graph.json',graph([('constant',[],dict(value=0)),('add',['arg0','0'],{})]));binary(base/'cst',[[1.]])
            args=SimpleNamespace(work_dir=base/'worker',graph=base/'graph.json',constants=base/'cst')
            reference=np.zeros((1,10),dtype=np.float64) if rejected else np.full((1,10),32.,dtype=np.float64)
            r24=dict(worker=dict(approximate_probes=[dict(approximate=reference.tolist()) for _ in range(3)]))
            source=dict(sample_ids=['train:0','train:1','train:2'])
            with ExitStack() as stack:
                for key,value in dict(EXPECTED_NODE_COUNT=3,EXPECTED_EDGE_COUNT=2,CONSTANT_COUNT=1,
                    OP_COUNTS=dict(input=1,constant=1,add=1)).items():stack.enter_context(mock.patch.object(math,key,value))
                stack.enter_context(mock.patch.object(math,'check_sources',return_value=({}, {},r24)))
                stack.enter_context(mock.patch.object(math.original,'_load_probes',return_value=(
                    np.zeros((3,3,32,32),dtype=np.float64),np.array([0,1,2]),source)))
                return args, math.worker(args)

        def test_three_synthetic_probes_use_actual_lazy_arithmetic_and_strict_checks(self):
            with tempfile.TemporaryDirectory() as directory:
                args,result=self._run_fixture(directory)
                self.assertEqual(result['status'],'ok');self.assertTrue(result['optimized_decoded_ideal_math_passed'])
                self.assertEqual((result['packed_forward_completed'],result['math_check_completed']),(3,3))
                self.assertEqual(result['op_completed'],6)
                self.assertEqual(result['constant_reads']['materialization_attempts'],3)
                self.assertTrue(all(row['math']['passed'] for row in result['probe_rows']))
                self.assertEqual(result['ckks_executions'],0)
                self.assertEqual(result['constant_reads']['persistent_numeric_cache_bytes'],0)
                self.assertTrue((args.work_dir/'constant-index.json').is_file())

        def test_strict_failure_preserves_first_probe_partial_and_stops_remaining(self):
            with tempfile.TemporaryDirectory() as directory:
                args,result=self._run_fixture(directory,rejected=True)
                self.assertEqual(result['status'],'error')
                self.assertEqual((result['packed_forward_completed'],result['math_check_completed']),(1,1))
                self.assertFalse(result['probe_rows'][0]['math']['passed'])
                self.assertFalse(result['optimized_decoded_ideal_math_passed'])
                saved=json.loads((args.work_dir/'worker-result.json').read_text())
                self.assertEqual(saved['packed_forward_attempts'],1)
    return MathCases


@require(NUMPY_ARTIFACT)
class EnvironmentCases(unittest.TestCase):
    def test_only_new_synthetic_numeric_cases(self):
        from pt_agent.experiments.r10_paper_metrics import run_process
        with tempfile.TemporaryDirectory(prefix='r33-tests-') as directory:
            out=Path(directory)
            observation=run_process([str(NUMPY_PYTHON),'-B','-m','pt_agent.tests.test_r33_orbit_optimized_math','--inner'],
                cwd=Path(__file__).resolve().parents[2],log=out/'tests.log',resources=out/'resources.jsonl',
                timeout=60,mem_gib=.5,fsize_gib=.1,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            log=(out/'tests.log').read_text();print(log,end='',flush=True)
            self.assertEqual(observation['exit_code'],0,log)
            self.assertFalse(observation['timed_out']);self.assertIsNone(observation['signal'])
            self.assertEqual(observation['remaining_pids'],[])
            self.assertIn('Ran 6 tests',log);self.assertNotIn('skipped',log)


if __name__=='__main__':
    if sys.argv[1:]==['--inner']:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(numeric_cases())
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() and result.testsRun==6 and not result.skipped else 1)
    unittest.main()
