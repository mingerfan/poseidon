"""R25 small saved-graph and binary fixtures; no official data are opened."""

from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class PackedCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        from pt_agent.experiments import r25_orbit_packed_worker as worker
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        cls.worker, cls.np, cls.Recorder = worker, worker._numeric_modules(), StageRecorder

    def setUp(self):
        temporary = TemporaryDirectory(prefix='r25-inner-', dir=os.environ.get('R25_TEST_WORK'))
        self.addCleanup(temporary.cleanup); self.case = Path(temporary.name)
        self.work = self.case / 'work/checks-worker'; self.work.mkdir(parents=True)
        self.recorder = self.Recorder(self.work / 'events.jsonl')
        self.args = SimpleNamespace(work_dir=self.work, graph=self.case/'fixture-graph.json',
            constants=self.case/'fixture.cst', probes=self.case/'fixture-probes',
            probe_source=self.case/'fixture-probe-source.json',
            r13_result=self.case/'r13.json', r22_result=self.case/'r22.json', r24_result=self.case/'r24.json')

    @staticmethod
    def _graph():
        return dict(inputs=['arg0'], outputs=['6'], nodes=[
            dict(id='arg0', op='input', op_descr={}),
            dict(id='0', op='constant', op_descr=dict(value=0)),
            dict(id='1', op='rotate', op_descr=dict(offset=2)),
            dict(id='2', op='rotate', op_descr=dict(offset=-1)),
            dict(id='3', op='mul', op_descr={}),
            dict(id='4', op='add', op_descr={}),
            dict(id='5', op='negate', op_descr={}),
            dict(id='6', op='add', op_descr={})], edges=[
                dict(source=source, target=target) for source, target in
                [('arg0','1'), ('arg0','2'), ('0','3'), ('1','3'),
                 ('2','4'), ('3','4'), ('4','5'), ('0','6'), ('5','6')]])

    def _program(self, record=None):
        self.args.graph.write_text(json.dumps(self._graph() if record is None else record))
        return self.worker.load_program(self.args.graph)

    def _binary(self, entries, *, count=None, trailing=b''):
        payload = struct.pack('<q', len(entries) if count is None else count)
        for values in entries:
            payload += struct.pack('<q', len(values))
            payload += struct.pack('<'+'d'*len(values), *values)
        self.args.constants.write_bytes(payload+trailing)
        return self.args.constants

    def _execute(self, program=None, constants=None, packed=None, result=None):
        program = self._program() if program is None else program
        if constants is None:
            self._binary([[2.0,3.0]])
            constants, _ = self.worker.read_constants(self.args.constants, 8, 1)
        packed = self.np.arange(8, dtype=self.np.float64) if packed is None else packed
        result = self.worker._initial_result() if result is None else result
        with patch.object(self.worker.shutil, 'disk_usage', return_value=SimpleNamespace(free=11*1024**3)):
            output = self.worker.execute_program(program, constants, packed, self.args, self.recorder, result)
        return output, result, constants, packed

    def test_graph_keeps_native_order_incoming_indices_and_last_use(self):
        program = self._program()
        self.assertEqual([node['id'] for node in program['nodes']], ['arg0']+[str(i) for i in range(7)])
        self.assertEqual(program['incoming']['6'], ['0','5'])
        self.assertEqual(program['last_use']['arg0'], 3)
        self.assertEqual(program['metadata']['node_count'], 8)
        self.assertEqual(program['metadata']['edge_count'], 9)
        self.assertEqual(program['metadata']['op_counts'],
            dict(input=1, constant=1, rotate=2, mul=1, add=2, negate=1))
        self.assertEqual(program['metadata']['static_terminal_ids'], ['6'])

    def test_graph_rejects_duplicate_ids_edges_forward_references_and_noncontinuous_ids(self):
        variants = []
        record = self._graph(); record['nodes'][2]['id']='0'; variants.append(record)
        record = self._graph(); record['edges'].append(copy.deepcopy(record['edges'][0])); variants.append(record)
        record = self._graph(); record['edges'][0]['source']='6'; variants.append(record)
        record = self._graph(); record['nodes'][2]['id']='9'; variants.append(record)
        record = self._graph(); record['edges'][0]['source']='missing'; variants.append(record)
        for record in variants:
            with self.subTest(record=record), self.assertRaises(self.worker.WorkerFailure):
                self._program(record)

    def test_graph_rejects_arity_unsupported_ops_and_invalid_descriptors(self):
        variants = []
        record = self._graph(); record['edges'].pop(); variants.append(record)
        record = self._graph(); record['nodes'][5]['op']='bootstrap'; variants.append(record)
        record = self._graph(); record['nodes'][1]['op_descr']['value']='0'; variants.append(record)
        record = self._graph(); record['nodes'][1]['op_descr']['value']=-1; variants.append(record)
        record = self._graph(); record['nodes'][1]['op_descr']['value']=True; variants.append(record)
        record = self._graph(); record['nodes'][2]['op_descr']['offset']=0.5; variants.append(record)
        record = self._graph(); record['nodes'][2]['op_descr']['offset']=False; variants.append(record)
        record = self._graph(); record['inputs']=[]; variants.append(record)
        record = self._graph(); record['outputs']=['5']; variants.append(record)
        for record in variants:
            with self.subTest(record=record), self.assertRaises(self.worker.WorkerFailure):
                self._program(record)

    def test_constants_repeat_positive_lengths_and_keep_one_readonly_array_per_id(self):
        path = self._binary([[2.0, -3.0, 5.0], [7.0]])
        arrays, metadata = self.worker.read_constants(path, 8, 2)
        self.assertEqual(set(arrays), {0,1})
        self.np.testing.assert_array_equal(arrays[0], [2,-3,5,2,-3,5,2,-3])
        self.np.testing.assert_array_equal(arrays[1], [7]*8)
        self.assertTrue(all(array.dtype == self.np.float64 and not array.flags.writeable for array in arrays.values()))
        self.assertEqual((metadata['count'], metadata['slot_count'], metadata['cache_nbytes']), (2,8,128))
        self.assertTrue(metadata['readonly'])
        self.assertEqual([row['raw_length'] for row in metadata['entries']], [3,1])
        with self.assertRaises(ValueError): arrays[0][0]=123.0

    def test_constants_truncate_long_payload_without_reading_ignored_nonfinite_tail(self):
        path = self._binary([[1.,2.,3.,4.,float('nan'),float('inf')], [9.]])
        arrays, metadata = self.worker.read_constants(path, 4, 2)
        self.np.testing.assert_array_equal(arrays[0], [1,2,3,4])
        self.np.testing.assert_array_equal(arrays[1], [9]*4)
        self.assertEqual(metadata['entries'][0]['prefix_values'],4)
        self.assertEqual(metadata['entries'][0]['skipped_values'],2)

    def test_constants_reject_count_nonpositive_length_truncation_bounds_and_effective_nonfinite(self):
        variants = [struct.pack('<q',0), struct.pack('<q',-1), struct.pack('<q',2),
            struct.pack('<qq',1,0), struct.pack('<qq',1,-1),
            struct.pack('<qq',1,1000000)+struct.pack('<d',1),
            struct.pack('<qq',1,2)+struct.pack('<d',1),
            struct.pack('<qqd',1,1,float('nan')), struct.pack('<qqd',1,1,float('inf'))]
        for payload in variants:
            self.args.constants.write_bytes(payload)
            with self.subTest(payload_bytes=len(payload)), self.assertRaises(self.worker.WorkerFailure):
                self.worker.read_constants(self.args.constants, 8, 1)

    def test_constants_open_once_and_seek_over_long_tail(self):
        path = self._binary([[float(i) for i in range(16)], [3.]])
        original = Path.open
        opens = []
        def opened(candidate, *args, **kwargs):
            if candidate == path: opens.append((args,kwargs))
            return original(candidate,*args,**kwargs)
        with patch.object(Path,'open',opened):
            arrays, metadata = self.worker.read_constants(path, 4, 2)
        self.assertEqual(len(opens),1)
        self.assertEqual(metadata['entries'][0]['skipped_values'],12)
        self.np.testing.assert_array_equal(arrays[1],[3]*4)

    def test_all_five_ops_positive_negative_rotation_and_non_inplace_values(self):
        output, result, constants, packed = self._execute()
        expected = -(self.np.roll(packed,1)+constants[0]*self.np.roll(packed,-2))+constants[0]
        self.np.testing.assert_array_equal(output,expected)
        self.np.testing.assert_array_equal(packed,self.np.arange(8,dtype=self.np.float64))
        self.np.testing.assert_array_equal(constants[0],[2,3,2,3,2,3,2,3])
        self.assertFalse(self.np.shares_memory(output,packed))
        self.assertFalse(self.np.shares_memory(output,constants[0]))
        self.assertEqual((result['op_attempts'],result['op_completed']),(7,7))
        self.assertEqual(result['op_attempted_by_kind'],dict(constant=1,rotate=2,mul=1,add=2,negate=1))
        self.assertEqual(result['op_completed_by_kind'],result['op_attempted_by_kind'])

    def test_last_use_matches_static_peak_and_each_execution_has_an_independent_pool(self):
        program=self._program(); first,result,constants,packed=self._execute(program=program)
        original=first.copy()
        second,result,_,_=self._execute(program=program,constants=constants,packed=packed+1,result=result)
        self.np.testing.assert_array_equal(first,original)
        self.assertFalse(self.np.shares_memory(first,second))
        self.assertEqual(result['observed_pool_peak'],program['metadata']['static_live_peak'])
        self.assertEqual((result['op_attempts'],result['op_completed']),(14,14))
        self.assertEqual([row['terminal_ids'] for row in result['pool_rows']],[['6'],['6']])
        self.assertTrue(all(row['pool_is_new'] for row in result['pool_rows']))
        self.assertEqual(result['current_pool_count'],1)

    def test_dead_results_are_executed_then_released(self):
        record=dict(inputs=['arg0'],outputs=['2'],nodes=[dict(id='arg0',op='input',op_descr={}),
            dict(id='0',op='constant',op_descr=dict(value=0)),dict(id='1',op='negate',op_descr={}),
            dict(id='2',op='rotate',op_descr=dict(offset=1))],
            edges=[dict(source='arg0',target='1'),dict(source='0',target='2')])
        program=self._program(record);output,result,constants,_=self._execute(program=program)
        self.np.testing.assert_array_equal(output,self.np.roll(constants[0],-1))
        self.assertEqual(program['metadata']['static_terminal_ids'],['2'])
        self.assertEqual(result['op_completed_by_kind']['negate'],1)

    def test_execution_rejects_missing_constant_and_nonfinite_intermediate_with_partial_counts(self):
        with self.assertRaises(self.worker.WorkerFailure):self._execute(constants={})
        packed=self.np.full(8,self.np.finfo(self.np.float64).max,dtype=self.np.float64)
        result=self.worker._initial_result()
        with self.np.errstate(over='ignore'),self.assertRaises(self.worker.WorkerFailure):
            self._execute(packed=packed,result=result)
        self.assertGreater(result['op_attempts'],0)
        self.assertGreaterEqual(result['op_attempts'],result['op_completed'])

    def test_pack_and_decode_keep_exact_rgb_order_and_single_factor_32(self):
        image=self.np.arange(3072,dtype=self.np.float64).reshape(3,32,32);before=image.copy()
        packed=self.worker.pack_input(image)
        expected=self.np.tile(self.np.pad(image.reshape(-1)/32,(0,1024)),16)
        self.np.testing.assert_array_equal(packed,expected)
        self.np.testing.assert_array_equal(self.worker.pack_input(image[None]),expected)
        self.np.testing.assert_array_equal(image,before)
        self.assertEqual((packed.shape,packed.dtype),((65536,),self.np.dtype('float64')))
        self.np.testing.assert_array_equal(self.worker.decode_output(packed),image.reshape(-1)[:10].reshape(1,10))

    def test_pack_decode_and_math_reject_wrong_shape_dtype_and_nonfinite(self):
        for image in (self.np.zeros((3,32,32),dtype=self.np.float32),
                      self.np.zeros((3,16,32),dtype=self.np.float64),
                      self.np.full((3,32,32),self.np.nan,dtype=self.np.float64)):
            with self.subTest(shape=image.shape,dtype=image.dtype),self.assertRaises(self.worker.WorkerFailure):
                self.worker.pack_input(image)
        for packed in (self.np.zeros(65536,dtype=self.np.float32),
                       self.np.zeros(10,dtype=self.np.float64),
                       self.np.full(65536,self.np.nan,dtype=self.np.float64)):
            with self.subTest(shape=packed.shape,dtype=packed.dtype),self.assertRaises(self.worker.WorkerFailure):
                self.worker.decode_output(packed)
        reference=self.np.zeros((1,10),dtype=self.np.float64)
        self.assertTrue(self.worker._check_math(reference,reference.copy())['passed'])
        self.assertFalse(self.worker._check_math(reference,self.np.ones((1,10),dtype=self.np.float64)*1e-8)['passed'])
        for observed in (self.np.zeros((10,),dtype=self.np.float64),
                         self.np.zeros((1,10),dtype=self.np.float32),
                         self.np.full((1,10),self.np.inf,dtype=self.np.float64)):
            with self.subTest(shape=observed.shape,dtype=observed.dtype),self.assertRaises(self.worker.WorkerFailure):
                self.worker._check_math(reference,observed)

    def _run(self, *, failed_probe=None):
        self.args.graph.write_text(json.dumps(self._graph()))
        self._binary([[2.0]])
        images=self.np.stack([self.np.full((3,32,32),float(i),dtype=self.np.float64) for i in range(3)])
        labels=self.np.array([6,9,9],dtype=self.np.int64)
        source=dict(split='train',indices=[0,1,2],sample_ids=['train:0','train:1','train:2'],labels=labels.tolist())
        references=[dict(probe_index=i,sample_id='train:'+str(i),
            approximate=[[64.0-2*i]+[64.0-3*i]*9]) for i in range(3)]
        if failed_probe is not None: references[failed_probe]['approximate'][0][0]+=0.25
        r24=dict(worker=dict(approximate_probes=references))
        expected=dict(node_count=8,edge_count=9,output='6',static_live_peak=4,static_terminal_ids=['6'],
            op_counts=dict(input=1,constant=1,rotate=2,mul=1,add=2,negate=1))
        result=self.worker._initial_result(); self.last_result=result
        with ExitStack() as stack:
            checked=stack.enter_context(patch.object(self.worker,'_check_sources',return_value=({'fixture':True},{},{},r24)))
            probes=stack.enter_context(patch.object(self.worker,'_load_probes',return_value=(images,labels,source)))
            graph=stack.enter_context(patch.object(self.worker,'load_program',wraps=self.worker.load_program))
            constants=stack.enter_context(patch.object(self.worker,'read_constants',wraps=self.worker.read_constants))
            stack.enter_context(patch.object(self.worker,'EXPECTED_PROGRAM',expected))
            stack.enter_context(patch.object(self.worker,'EXPECTED_CONSTANT_COUNT',1))
            stack.enter_context(patch.object(self.worker.shutil,'disk_usage',return_value=SimpleNamespace(free=11*1024**3)))
            self.worker.run_checks(self.args,self.recorder,result)
        return result,dict(checked=checked,probes=probes,graph=graph,constants=constants)

    def test_synthetic_run_loads_once_executes_three_independent_pools_and_keeps_limited_scope(self):
        result,hooks=self._run()
        for hook in hooks.values():hook.assert_called_once()
        for prefix in ('graph_load','constant_load','probe_load'):
            self.assertEqual((result[prefix+'_attempts'],result[prefix+'_completed']),(1,1))
        for prefix in ('packed_forward','math_check'):
            self.assertEqual((result[prefix+'_attempts'],result[prefix+'_completed']),(3,3))
        self.assertEqual((result['op_attempts'],result['op_completed']),(21,21))
        self.assertEqual(result['op_completed_by_kind'],dict(constant=3,rotate=6,mul=3,add=6,negate=3))
        self.assertEqual(result['op_attempted_by_kind'],result['op_completed_by_kind'])
        self.assertEqual([row['sample_id'] for row in result['probe_rows']],['train:0','train:1','train:2'])
        self.assertTrue(all(row['passed'] and row['max_absolute_error']==0 for row in result['probe_rows']))
        self.assertTrue(result['source_checked']);self.assertTrue(result['native_packed_equivalence_verified'])
        self.assertEqual(result['native_packed_equivalence_scope'],
            'R22 original unoptimized packed graph on train:0,train:1,train:2; finite-input numerical agreement only')
        self.assertTrue(result['constants_remained_readonly'])
        self.assertTrue(all(row['terminal_ids']==['6'] for row in result['pool_rows']))
        self.assertTrue(all(result[key]==0 for key in self.worker.ZERO_FIELDS))
        self.assertTrue(all(result[key] is False for key in self.worker.FALSE_FIELDS))
        self.assertTrue(all(result[key] is None for key in self.worker.UNMEASURED_FIELDS))

    def test_first_math_failure_preserves_partial_record_and_leaves_third_probe_unstarted(self):
        with self.assertRaises(self.worker.WorkerFailure) as failure:self._run(failed_probe=1)
        self.assertEqual(failure.exception.status,'math_failed')
        result=self.last_result
        self.assertEqual((result['packed_forward_attempts'],result['packed_forward_completed']),(2,2))
        self.assertEqual((result['math_check_attempts'],result['math_check_completed']),(2,2))
        self.assertEqual((result['op_attempts'],result['op_completed']),(14,14))
        self.assertEqual([row['passed'] for row in result['probe_rows']],[True,False])
        self.assertTrue(result['source_checked']);self.assertFalse(result['native_packed_equivalence_verified'])
        saved=json.loads((self.work/'phase_result.json').read_text())
        self.assertEqual(saved['active_sample_id'],'train:1')
        self.assertEqual(len(saved['probe_rows']),2)
        self.assertFalse(saved['probe_rows'][1]['passed'])


if __name__ == '__main__':
    unittest.main()
