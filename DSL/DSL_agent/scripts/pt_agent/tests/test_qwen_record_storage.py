"""Detached metadata fixtures only; no model, arithmetic backend or call replay."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from pt_agent.experiments.qwen_record_storage import (
    InternedStages, MAX_ID, PackedCalls, install_empty_record_storage)


ROWS = [((), 0, ()), ((0,), 1, ()), ((0, None), 2, (1,)),
        ((None, 2), 3, (0, 1, 2)), ((3, 3), 4, ()), ((MAX_ID,), MAX_ID, (MAX_ID,))]


def pack(rows):
    result=PackedCalls()
    for row in rows:result.append(row)
    return result


def owned_tuple_containers(values):
    """Conservative recorder-container cost, excluding borrowed scalar nodes."""
    seen=set()
    def visit(value):
        if not isinstance(value,(list,tuple)) or id(value)in seen:return 0
        seen.add(id(value))
        return sys.getsizeof(value)+sum(visit(child)for child in value)
    return visit(values)


def synthetic_storage_measurement(count=20000):
    # These are synthetic integer metadata, never actual graph/call contents.
    nodes=list(range(count+2));rows=[];packed=PackedCalls();stages=[];interned=InternedStages()
    for index in range(count):
        operands=(() if index%5==0 else (nodes[index],)if index%5==1 else(nodes[index],None))
        extra=(nodes[index],nodes[index+1])if index%7==0 else()
        row=(operands,nodes[index+1],extra);rows.append(row);packed.append(row)
        pair=(index//1024,index%3);stages.append(pair);interned.append(pair)
    return dict(records=count,calls=dict(original_owned_containers=owned_tuple_containers(rows),
        packed_owned_capacity=packed.storage_bytes(),packed_payload_bytes=packed.payload_bytes()),
        stages=dict(original_owned_containers=owned_tuple_containers(stages),interned_owned_capacity=interned.storage_bytes(),
            distinct_pairs=len(interned._pairs)),values_equal=list(packed)==rows and interned==stages)


class ExactStorageChecks(unittest.TestCase):
    def test_all_shapes_none_duplicates_large_identifiers_and_order(self):
        compact=pack(ROWS)
        self.assertEqual(list(compact),ROWS);self.assertEqual(len(compact),len(ROWS))
        for index,row in enumerate(ROWS):self.assertEqual(compact[index],row)
        self.assertIsNone(compact[2][0][1]);self.assertEqual(compact[-1],ROWS[-1])

    def test_list_slicing_and_bidirectional_equality_match(self):
        compact=pack(ROWS)
        for selection in (slice(None),slice(None,None,-1),slice(1,5,2),slice(-3,None),slice(10,20)):
            self.assertEqual(compact[selection],ROWS[selection]);self.assertIs(type(compact[selection]),list)
        self.assertEqual(compact,ROWS);self.assertEqual(ROWS,compact);self.assertEqual(compact,pack(ROWS))
        self.assertNotEqual(compact,ROWS[::-1]);self.assertNotEqual(ROWS[::-1],compact)
        self.assertNotEqual(compact,tuple(ROWS))
        for index in (-7,6):
            with self.assertRaises(IndexError):compact[index]
        with self.assertRaises(ValueError):compact[::0]

    def test_append_rejects_lossy_or_malformed_values_without_partial_change(self):
        compact=pack(ROWS);before=list(compact);bytes_before=compact.payload_bytes()
        invalid=[([],0,()),((),True,()),((False,),0,()),((1.0,),0,()),((None,),-1,()),
                 ((0,),0,(None,)),((0,),MAX_ID+1,()),((MAX_ID+1,),0,()),((0,),0,(MAX_ID+1,)),
                 ((0,),0,[]),[(),0,()],((),0)]
        for row in invalid:
            with self.subTest(row=row),self.assertRaises((ValueError,TypeError)):compact.append(row)
            self.assertEqual(list(compact),before);self.assertEqual(compact.payload_bytes(),bytes_before)

    def test_clear_releases_capacity_and_keeps_previously_read_values(self):
        compact=pack(ROWS);saved=compact[:];before=compact.storage_bytes();compact.clear()
        self.assertEqual(list(compact),[]);self.assertEqual(saved,ROWS)
        self.assertLess(compact.storage_bytes(),before)
        compact.append(ROWS[-1]);self.assertEqual(list(compact),[ROWS[-1]])

    def test_stage_pairs_share_identity_preserve_list_values_and_clear(self):
        stages=InternedStages();expected=[]
        for index in range(100):
            pair=(index%3,index%2);expected.append(pair);stages.append(pair)
        self.assertEqual(stages,expected);self.assertEqual(stages[::-3],expected[::-3])
        self.assertIs(stages[0],stages[6]);self.assertEqual(len(stages._pairs),6)
        with self.assertRaises(ValueError):stages.append((False,0))
        snapshot=stages[:];stages.clear();self.assertEqual(stages,[]);self.assertEqual(stages._pairs,{})
        self.assertEqual(snapshot,expected)

    def test_only_fresh_original_record_factory_is_admitted(self):
        recorder=SimpleNamespace(policy='record',call_count=0,calls=[],node_stage=[],output_nodes=None)
        self.assertIs(install_empty_record_storage(recorder),recorder)
        self.assertIsInstance(recorder.calls,PackedCalls);self.assertIsInstance(recorder.node_stage,InternedStages)
        with self.assertRaises(ValueError):install_empty_record_storage(recorder)
        for changed in (dict(policy='plan'),dict(call_count=1),dict(calls=[ROWS[0]]),dict(node_stage=[(-1,0)]),dict(output_nodes=[])):
            original=dict(policy='record',call_count=0,calls=[],node_stage=[],output_nodes=None);original.update(changed)
            candidate=SimpleNamespace(**original)
            with self.subTest(changed=changed),self.assertRaises(ValueError):install_empty_record_storage(candidate)
            self.assertIs(candidate.calls,original['calls'])

    def consumer(self,relative,name,namespace):
        path=Path(__file__).parents[1]/relative;tree=ast.parse(path.read_text())
        selected=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name==name)
        exec(compile(ast.Module(body=[selected],type_ignores=[]),str(path),'exec'),namespace)
        return namespace[name]

    def test_original_express_body_reads_identical_calls_without_replay(self):
        namespace=dict(PlanError=ValueError,PLAN_FORMAT='pt-agent-refresh-plan-v1',
            levels=lambda profile:dict(input_level=5,boot_in_level=1,boot_out_level=5))
        self.consumer('plan.py','_ranges',namespace);express=self.consumer('plan.py','express',namespace)
        inner=SimpleNamespace(kind=['input','input','neg','add'],args=[(),(),(1,),(0,2)])
        rows=[((),0,()),((),1,()),((0,1),3,(2,))]
        placement=SimpleNamespace(refreshed_uses={2:{3}},boot_nodes=[2],boots=1,barriers=[],cuts=[],latency_us=17)
        old=express(inner,SimpleNamespace(calls=rows,output_nodes=[3],call_count=3),placement)
        new=express(inner,SimpleNamespace(calls=pack(rows),output_nodes=[3],call_count=3),placement)
        self.assertEqual(new,old);self.assertEqual(new['substitutions'],[[2,1,2]])

    def test_original_stage_labels_body_reads_interned_pairs_exactly(self):
        labels=self.consumer('stage_plan.py','labels',dict(StageOrderError=ValueError,MUL={'mul_cc','mul_cp'}))
        dag=SimpleNamespace(kind=['input','mul_cc','mul_cc'],args=[(),(0,0),(1,1)])
        pairs=[(-1,0),(0,1),(0,1)];interned=InternedStages()
        for pair in pairs:interned.append(tuple(pair))
        self.assertEqual(labels(dag,SimpleNamespace(node_stage=pairs)),labels(dag,SimpleNamespace(node_stage=interned)))

    def test_small_retention_measurement_compares_owned_containers_only(self):
        record=synthetic_storage_measurement(20000)
        self.assertTrue(record['values_equal'])
        self.assertLess(record['calls']['packed_owned_capacity'],record['calls']['original_owned_containers'])
        self.assertLess(record['stages']['interned_owned_capacity'],record['stages']['original_owned_containers'])


if __name__=='__main__':unittest.main()
