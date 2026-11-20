"""Draft loader-only dispatch checks; no model, planner, Count or native run."""
import ast
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_component_cost as helper
from pt_agent.experiments import qwen_layer_native_probe as probe
from pt_agent.experiments import qwen_remaining_prefill_compile as remaining
from pt_agent.experiments import qwen_layered_structure as layered
from pt_agent.tests.test_qwen_layered_structure import fixture


PROBE_PREIMAGE = '''def _load(args,result):
    info,program=original._load_program(args,result)
    partition=layered.split_program(program,{name:array.shape for name,array in info.graph.weights.items()})
    callback=layered.callbacks(partition,info.graph.weights)[0][0]
    return partition,callback'''
REMAINING_PREIMAGE = '''def _load(args,result):
    info,program=original._load_program(args,result)
    partition=layered.split_program(program,{name:array.shape for name,array in info.graph.weights.items()})
    layers,head=layered.callbacks(partition,info.graph.weights)
    callback=head if args.component=='head'else layers[spec(args.component)['index']]
    if layered.source_binding(callback.segment)!=result['source']['component_source']:
        raise ValueError('Actual loaded original component differs from the admitted frozen source')
    return callback'''


class ComponentLoaderScopedChecks(unittest.TestCase):
    def setUp(self):
        # Existing public fixture parses shape-only DSL. No array/model is loaded.
        self.program,self.shapes=fixture()
        self.weights={k:SimpleNamespace(shape=tuple(v))for k,v in self.shapes.items()}
        self.info=SimpleNamespace(graph=SimpleNamespace(weights=self.weights))
        self.partition=layered.split_program(self.program,self.shapes)

    def remaining_arguments(self,name):
        segment=self.partition.head if name=='head'else self.partition.layers[remaining.spec(name)['index']]
        return SimpleNamespace(component=name),dict(source=dict(component_source=layered.source_binding(segment)))

    def test_probe_actual_loader_wraps_original_first_callback_once(self):
        args=SimpleNamespace();result={}
        with patch.object(probe.original,'_load_program',return_value=(self.info,self.program))as load, \
                patch.object(helper,'already_scoped_lowering',wraps=helper.already_scoped_lowering)as adapt:
            partition,callback=probe._load(args,result)
        load.assert_called_once_with(args,result);adapt.assert_called_once()
        original=adapt.call_args.args[0]
        self.assertIs(callback.segment,partition.layers[0])
        self.assertIs(callback.segment,original.segment)
        self.assertIs(callback.weights,self.weights)
        self.assertIs(callback.weights,original.weights)
        self.assertIsInstance(callback,layered.SegmentLowering)

    def test_remaining_actual_loader_preserves_layer_and_head_selection(self):
        for name in ('layer1','layer23','head'):
            with self.subTest(component=name):
                args,result=self.remaining_arguments(name)
                with patch.object(remaining.original,'_load_program',return_value=(self.info,self.program))as load, \
                        patch.object(helper,'already_scoped_lowering',wraps=helper.already_scoped_lowering)as adapt:
                    callback=remaining._load(args,result)
                load.assert_called_once_with(args,result);adapt.assert_called_once()
                original=adapt.call_args.args[0]
                self.assertIs(callback.segment,original.segment)
                self.assertIs(callback.weights,self.weights)
                self.assertEqual(callback.segment.role,'head'if name=='head'else'layer')
                self.assertEqual(callback.segment.index,None if name=='head'else int(name[5:]))
                self.assertEqual(layered.source_binding(callback.segment),result['source']['component_source'])

    def test_remaining_source_rejection_precedes_adapter(self):
        args,result=self.remaining_arguments('layer1');result['source']['component_source']={}
        with patch.object(remaining.original,'_load_program',return_value=(self.info,self.program)), \
                patch.object(helper,'already_scoped_lowering',side_effect=AssertionError('adapter before original source check'))as adapt:
            with self.assertRaises(ValueError):remaining._load(args,result)
        adapt.assert_not_called()

    def test_original_load_error_identity_precedes_adapter(self):
        for module,args,result in ((probe,SimpleNamespace(),{}),(remaining,*self.remaining_arguments('head'))):
            with self.subTest(module=module.__name__):
                error=InterruptedError('original loader fixture')
                with patch.object(module.original,'_load_program',side_effect=error), \
                        patch.object(helper,'already_scoped_lowering',side_effect=AssertionError('adapter after failed load'))as adapt:
                    with self.assertRaises(InterruptedError)as caught:module._load(args,result)
                self.assertIs(caught.exception,error);adapt.assert_not_called()

    def test_loader_adapter_checks_original_weight_shape_before_lower(self):
        for name in ('layer1','head'):
            with self.subTest(component=name):
                args,result=self.remaining_arguments(name)
                with patch.object(remaining.original,'_load_program',return_value=(self.info,self.program)):
                    callback=remaining._load(args,result)
                key,shape=callback.segment.weight_shapes[0];original=self.weights[key]
                self.weights[key]=SimpleNamespace(shape=(*shape,1))
                try:
                    with patch('pt_agent.lower.lower',side_effect=AssertionError('arithmetic entered before weight validation'))as lower:
                        with self.assertRaises(ValueError):callback(object())
                    lower.assert_not_called()
                finally:self.weights[key]=original

    def test_inverse_AST_recovers_entire_original_loader_functions(self):
        expected_import=ast.parse('from .qwen_component_cost import already_scoped_lowering').body[0]
        expected_assign=ast.parse('callback=already_scoped_lowering(callback)').body[0]
        for function,preimage in ((probe._load,PROBE_PREIMAGE),(remaining._load,REMAINING_PREIMAGE)):
            tree=ast.parse(inspect.getsource(function));body=tree.body[0].body
            self.assertEqual(ast.dump(body[-3],include_attributes=False),ast.dump(expected_import,include_attributes=False))
            self.assertEqual(ast.dump(body[-2],include_attributes=False),ast.dump(expected_assign,include_attributes=False))
            del body[-3:-1]
            self.assertEqual(ast.dump(tree,include_attributes=False),ast.dump(ast.parse(preimage),include_attributes=False))


if __name__=='__main__':unittest.main()
