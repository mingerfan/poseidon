"""Finite scalar planner/export protocol fixtures; no model or Count execution."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pt_agent.experiments import qwen_component_cost as cost


def row(name, boundary, internal_boots=2, *, index=None):
    counts=dict(add=4,mul_cp=3,rot=2,boot=internal_boots)
    return dict(name=name,role='head'if name=='head'else'layer',index=index,
        slots=64,caller_connector_blocks=3,output_shape=[1,1,boundary*32],
        source=dict(original_ops=[dict(id=7,lineno=12,call_lines=[90])]),
        statements=[dict(id=7,latency_us=40.,boots=internal_boots)],
        boundary_boots=boundary,output_levels=[5]*boundary,
        internal=dict(counts=counts,boots_pred=internal_boots,latency_pred_us=100.,max_depth=4,hotspots=[]),
        complete_count=dict(counts=dict(add=8,mul_cp=3,rot=2,boot=internal_boots+boundary),rot_steps=[1,2]),
        constants=('shared','constant-'+name),plan_artifact=name+'/refresh-plan.json')


class ComponentCostChecks(unittest.TestCase):
    def test_aggregate_derived_boundaries_and_kernel_scope(self):
        rows=[row('layer0',2,index=0),row('layer1',2,index=1),row('head',1)]
        result=cost.aggregate(rows,boot_us=10.)
        self.assertEqual(result['explicit_boundary_boots'],5)
        self.assertEqual(result['component_internal_boots'],6)
        self.assertEqual(result['boots_pred'],11)
        self.assertEqual(result['latency_pred_us'],350.)
        projection=result['full_generation_projection']
        self.assertEqual(projection['total_boundary_boot_requests'],13)
        self.assertEqual(projection['component_calls'],8)
        self.assertEqual(projection['unique_components'],5)
        self.assertIsNone(projection['complete_boot_inventory'])
        self.assertIsNone(result['compiled_bootstraps'])
        self.assertFalse(result['global_optimality'])
        self.assertEqual(rows[0]['output_levels'],[5,5])

    def test_original_cse_and_complete_raw_counts_remain_distinct(self):
        result=cost.aggregate([row('layer0',2,index=0),row('head',1)],boot_us=10.)
        self.assertEqual(result['counts']['add'],8)
        self.assertEqual(result['complete_count_replay_counts']['add'],16)
        self.assertEqual(result['rot_keys'],2)
        self.assertEqual(result['distinct_consts'],3)
        self.assertEqual(result['const_bytes'],3*64*8)
        self.assertEqual(result['max_depth'],4)
        self.assertEqual(result['boot_nodes'],[])
        self.assertFalse(result['global_boot_node_ids_available'])

    def test_output_level_missing_boot_or_foreign_geometry_rejected(self):
        rows=[row('layer0',2,index=0),row('head',1)]
        for change in ('level','boot','slots','connector','order'):
            modified=copy.deepcopy(rows)
            if change=='level':modified[0]['output_levels'][0]=4
            elif change=='boot':modified[0]['complete_count']['counts']['boot']-=1
            elif change=='slots':modified[0]['slots']=128
            elif change=='connector':modified[0]['caller_connector_blocks']=4
            else:modified[0]['name']='layer1'
            with self.subTest(change=change),self.assertRaises(ValueError):cost.aggregate(modified,boot_us=10.)

    def test_source_hotspot_mapping_and_boundary_cost_are_explicit(self):
        result=cost.aggregate([row('layer0',2,index=0),row('head',1)],boot_us=10.)
        sources=[h for h in result['hotspots']if h['line']==12]
        self.assertEqual(len(sources),1)
        self.assertEqual(sources[0]['call_lines'],[90])
        self.assertEqual(sources[0]['boots'],4)
        boundaries=[h for h in result['hotspots']if h['scope']=='component-boundary']
        self.assertEqual(boundaries[0]['line'],None)
        self.assertEqual(boundaries[0]['boots'],3)

    def test_serial_estimator_writes_completed_protocol_only(self):
        sequence=[]
        callbacks=[SimpleNamespace(segment=SimpleNamespace(role='layer',index=0)),
            SimpleNamespace(segment=SimpleNamespace(role='head',index=None))]
        class API:
            boot_us=10.
            def split(self,program,weights):
                self_program=program;self_weights=weights
                if self_program!='candidate' or self_weights!='original aliases':raise AssertionError('source changed')
                return callbacks
            def measure(self,callback):
                name=callback.segment.role+(''if callback.segment.index is None else str(callback.segment.index))
                sequence.append(name);return row(name,2 if name=='layer0'else 1,index=callback.segment.index)
        with tempfile.TemporaryDirectory()as temp:
            directory=Path(temp)/'new'
            result=cost.estimate('candidate','original aliases',directory=directory,api=API())
            saved=json.loads((directory/'result.json').read_text())
            self.assertEqual(sequence,['layer0','head'])
            self.assertEqual(saved['status'],'ok')
            self.assertEqual(saved['boots_pred'],result['boots_pred'])
            self.assertEqual(json.loads((directory/'partial.json').read_text())['completed'],2)

    def test_first_error_preserves_partial_and_stops_following_component(self):
        sequence=[];interrupted=InterruptedError('fixture stop')
        class API:
            boot_us=10.
            def split(self,*args):return [SimpleNamespace(segment=SimpleNamespace(role='layer',index=i))for i in range(3)]
            def measure(self,callback):
                index=callback.segment.index;sequence.append(index)
                if index==1:raise interrupted
                return row('layer0',2,index=0)
        with tempfile.TemporaryDirectory()as temp:
            directory=Path(temp)/'new'
            with self.assertRaises(InterruptedError)as caught:cost.estimate(None,None,directory=directory,api=API())
            self.assertIs(caught.exception,interrupted)
            self.assertEqual(sequence,[0,1])
            partial=json.loads((directory/'partial.json').read_text())
            self.assertEqual(partial['status'],'error');self.assertEqual(partial['completed'],1)
            self.assertFalse((directory/'result.json').exists())


if __name__=='__main__':unittest.main()
