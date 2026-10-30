"""U4 §7.10: memory accounting and immutable-type reuse preserve values."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import gc
import json
from pathlib import Path
import tempfile
import tracemalloc
import unittest

from pt_agent.manage.greedy import RecordingBackend, memory_state, timed_stage, timing_log
from pt_agent.manage.ir import Node, Type, shared_type


class ManagementMemoryTests(unittest.TestCase):
    def test_slots_and_copy_preserve_fields_and_private_attrs(self):
        value = Type(True, 40, 0)
        self.assertFalse(hasattr(value, '__dict__'))
        with self.assertRaises(FrozenInstanceError):
            value.level = 1
        left = Node(1, 'rotate', [-1], value, {'offset': [3]})
        right = replace(left, args=list(left.args), attrs=dict(left.attrs))
        self.assertFalse(hasattr(left, '__dict__'))
        self.assertEqual(left, right)
        self.assertIsNot(left.attrs, right.attrs)
        right.attrs['offset'] = [7]
        self.assertEqual(left.attrs, {'offset': [3]})
        self.assertEqual(deepcopy(left), left)

    def test_live_nodes_include_copies_and_are_released(self):
        gc.collect()
        start = Node.live_count
        node = Node(1, 'negate', [-1], Type(True, 40, 0))
        copied = deepcopy(node)
        self.assertEqual(Node.live_count, start+2)
        del node, copied
        gc.collect()
        self.assertEqual(Node.live_count, start)

    def test_recording_reuses_only_frozen_types(self):
        recorded = RecordingBackend(16)
        x = recorded.input(0)
        first, second = x+x, x+x
        self.assertEqual(first.__dict__, {})
        self.assertEqual(recorded.__dict__, {})
        a, b = recorded.nodes
        self.assertIs(a.type, b.type)
        self.assertIs(a.type, shared_type(True, 0, 0))
        self.assertIsNot(a.attrs, b.attrs)
        self.assertIsNot(a.args, b.args)
        self.assertEqual(first.nid, 0)
        self.assertEqual(second.nid, 1)

    def test_always_record_memory_without_enabling_tracemalloc(self):
        self.assertFalse(tracemalloc.is_tracing())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'stages.json'
            with timing_log(path), timed_stage('plain'):
                node = Node(1, 'negate', [-1], Type(True, 40, 0))
            row = json.loads(path.read_text())['stages'][0]
            for key in ('memory_enter', 'memory_exit'):
                self.assertGreater(row[key]['VmRSS_bytes'], 0)
                self.assertGreaterEqual(row[key]['VmHWM_bytes'], row[key]['VmRSS_bytes'])
            self.assertEqual(row['memory_exit']['node_instances'], row['memory_enter']['node_instances']+1)
            self.assertNotIn('allocations', row)
            self.assertFalse(tracemalloc.is_tracing())
            del node

    def test_diagnostic_snapshot_groups_allocations_by_line(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'stages.json'
            tracemalloc.start()
            try:
                with timing_log(path), timed_stage('diagnostic'):
                    data = bytearray(4096)
                row = json.loads(path.read_text())['stages'][0]
                self.assertGreaterEqual(row['allocations']['total_bytes'], len(data))
                self.assertLessEqual(len(row['allocations']['top20']), 20)
                self.assertTrue(all({'file', 'line', 'bytes', 'count'} <= set(item)
                                    for item in row['allocations']['top20']))
            finally:
                tracemalloc.stop()

    def test_failure_keeps_memory_and_shallow_table_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'stages.json'
            with timing_log(path), self.assertRaises(MemoryError):
                with timed_stage('failure'):
                    raise MemoryError('diagnostic fixture')
            row = json.loads(path.read_text())['stages'][0]
            self.assertFalse(row['complete'])
            self.assertIn('memory_exit', row)
            self.assertIn('MemoryError', row['error'])
            self.assertTrue(row['failure_objects'])
        state = memory_state()
        self.assertEqual(state['node_instances'], Node.live_count)


class PlanMemoryTests(unittest.TestCase):
    @staticmethod
    def plan():
        value=lambda name,kind,level: dict(id=name,kind=kind,level=level,scale_log2=43,components=2,ntt=True)
        return dict(values=[value('x','ciphertext',5),value('p','plaintext',5),value('y','ciphertext',4)],
                    external_inputs=['x'],final_outputs=['y'],
                    initialization=[dict(kind='init',op='encode',output='p',payload=dict(kind='inline',values=[.25,-1.5]))],
                    execution=[dict(kind='compute',op='mul_cp',inputs=['x','p'],output='y',attrs={'target':4})],
                    finalization=[dict(kind='store',inputs=['y'])])

    def test_compact_value_and_step_representation_is_exact(self):
        from pt_agent.experiments.u4_bert_managed import states,_state_tuples
        plan=self.plan()
        keys=('kind','level','scale_log2','components','ntt')
        expected=[(row['op'],tuple(tuple(v[k] for k in keys) for v in row['inputs']),
                   tuple(row['output'][k] for k in keys) if row['output'] is not None else None,row['attrs'])
                  for row in states(plan)]
        self.assertEqual(list(_state_tuples(plan)),expected)

    def test_full_comparator_preserves_success_and_all_failure_categories(self):
        from pt_agent.experiments.u4_bert_managed import compare_complete,compare_saved
        with tempfile.TemporaryDirectory() as temp:
            left,right=Path(temp)/'left.runtime-plan.json',Path(temp)/'right.runtime-plan.json'
            original=self.plan()
            def run(plan):
                left.write_text(json.dumps(original));right.write_text(json.dumps(plan))
                return compare_complete(left,right),compare_saved(left,right)
            complete,saved=run(original)
            self.assertTrue(complete['equal']);self.assertTrue(saved['equal'])
            self.assertEqual(complete['constant_comparisons'],[True])
            self.assertEqual(complete['bundle_constant_counts'],[0,0])
            for change in ('state','attrs','payload','length','missing_step','blob_count'):
                plan=deepcopy(original)
                if change=='state':plan['values'][-1]['level']=3
                elif change=='attrs':plan['execution'][0]['attrs']['target']=3
                elif change=='payload':plan['initialization'][0]['payload']['values'][-1]=-1.6
                elif change=='length':plan['initialization'].append(dict(kind='init',op='extra'))
                elif change=='missing_step':plan['finalization']=[]
                else:
                    folder=left.with_name('left.bundle');folder.mkdir()
                    (folder/'manifest.json').write_text(json.dumps(dict(blobs=[{}])))
                with self.subTest(change=change):
                    complete,saved=run(plan)
                    self.assertFalse(complete['equal'])
                    if change=='payload':
                        self.assertTrue(complete['step_states_equal'])
                        self.assertFalse(complete['constants_equal'])
                        self.assertFalse(saved['payload_descriptors_equal'])
                    elif change=='length':self.assertFalse(saved['payload_descriptors_equal'])
                    elif change=='blob_count':self.assertEqual(complete['bundle_constant_counts'],[1,0])
                    else:self.assertFalse(saved['step_states_equal'])


if __name__ == '__main__':
    unittest.main()
