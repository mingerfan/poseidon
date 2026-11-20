"""R16 environment-local tests with small tensors and temporary source records."""

import ast
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class _SmallCases:
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name]='1'
        from pt_agent.experiments import r16_baseline_validation_worker as worker
        worker._initialize_torch()
        import numpy as np
        import torch
        from pt_agent.experiments.r13_common_models import FACTORIES
        from pt_agent.data import cifar

        cls.np, cls.torch, cls.worker, cls.factories, cls.cifar = np, torch, worker, FACTORIES, cifar

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='r16-small-cases-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def _fixture(self, baseline='orbit', suffix='vgg16_silu_avgpool'):
        model='pt_agent.experiments.r13_common_models:'+suffix
        source_dir=self.root/suffix/'approved-source-fixture'
        source_dir.mkdir(parents=True)
        metadata=dict(self.factories[model],source=source_dir/'original.py',
                      checkpoint=source_dir/'checkpoint.pt',
                      inference_preprocessing_source=source_dir/'inference.py')
        for key in ('source','checkpoint','inference_preprocessing_source'):
            metadata[key].write_text('temporary provenance fixture; never executed or deserialized\n')
        factories=dict(self.factories);factories[model]=metadata
        r13=self.root/('r13-common-'+suffix.replace('_','-')+'-20261115')
        info_dir=r13/'worker/model_info'
        paths=dict(r13_result=r13/'result.json',model_info=info_dir/'model_info.json',
                   graph_record=info_dir/'graph/graph.json',probes=r13/'worker/training_probes.npz',
                   probe_source=r13/'worker/training_probe_source.json',
                   checkpoint=Path(metadata['checkpoint']),model_source=Path(metadata['source']),
                   baseline_root=self.root/baseline,work_dir=self.root/'validation/worker')
        normalization=dict(mean=list(self.cifar.MEAN),std=list(self.cifar.STD))
        indices,labels=[0,1,2],[6,9,9]
        source=dict(records=str(self.root/'raw-records'),selections=str(self.root/'selections.json'),
                    split='train',indices=indices,sample_ids=['train:'+str(i) for i in indices],
                    labels=labels,entries=[dict(index=i,label=label,shard='train-00000.records',row=i)
                                         for i,label in zip(indices,labels)],normalization=normalization,
                    dataset_metadata=dict(format='poseidon-cifar10-binary-records-v1',record_bytes=3073,
                        row_label_byte=0,preprocessing_applied=False,train_count=50000,test_count=10000,
                        pixel_layout='uint8 planar RGB 3x32x32, C order'))
        model_source=dict(type='torch',path=model,factory=model,spec_revision='R13',
                          original_source=str(metadata['source'].resolve()),
                          checkpoint=str(metadata['checkpoint'].resolve()),
                          inference_preprocessing_source=str(metadata['inference_preprocessing_source'].resolve()),
                          normalization=normalization,calibration_source=source,
                          preprocessing_provenance='official inference test',training_preprocessing_verified=False)
        split=dict(kind='legacy_cifar',heldout=list(self.cifar.HELDOUT),confirm=list(self.cifar.CONFIRM),
                   report=list(self.cifar.REPORT),calibration='frozen training selection')
        info=dict(source=model_source,graph='graph',input_handling=dict(internal_regression_source='legacy_cifar'),
                  inputs=[dict(name='original_features',kind='features',shape=[1,3,32,32],dtype='float64')],
                  sequence=None,task='classification',metric='classification_top1',threshold=dict(acc_drop_pp=0.5),
                  split=split,assumptions=[])
        graph=dict(name='small_interface',inputs=['original_features'],outputs=['original_logits'],nodes=[],
                   tensors=[['original_features',[1,3,32,32],'input'],['original_logits',[1,10],'act']])
        references=[(self.np.arange(10,dtype=self.np.float64).reshape(1,10)+i).tolist() for i in range(3)]
        math_samples=[dict(probe_index=i,passed=True,finite=True,shape_ok=True,input_name='original_features',
                           output_name='original_logits',reference_shape=[1,10],output_shape=[1,10],
                           reference=reference,output=reference,max_absolute_error=0.)
                      for i,reference in enumerate(references)]
        original=dict(round='R13',mode='common-model-intake-preparation',status='ok',model=model,
            worker=dict(round='R13',status='ok',model=model,import_completed=True,real_probes_passed=True,
                graph_math_passed=True,roundtrip_passed=True,static_coverage=True,model_source=model_source,
                split=split,source=source,graph_math=dict(passed=True,criterion='1e-12+1e-12*abs(reference)',
                samples=math_samples),artifacts=dict(model_info=str(info_dir),training_probes=str(paths['probes']),
                training_probe_source=str(paths['probe_source']))))
        for key,value in (('r13_result',original),('model_info',info),('graph_record',graph),('probe_source',source)):
            self._json(paths[key],value)
        images=self.np.broadcast_to(self.np.arange(3,dtype=self.np.float64).reshape(3,1,1,1),(3,3,32,32)).copy()
        self.np.savez(paths['probes'],input=images,labels=self.np.array(labels,dtype=self.np.int64),
                      indices=self.np.array(indices,dtype=self.np.int64))
        paths['baseline_root'].mkdir(exist_ok=True)
        if baseline=='orbit':
            source_names=dict(resnet20_silu='ResNetSiLU.py',vgg16_silu_avgpool='VGG16SiLU.py',
                              alexnet_silu_avgpool='AlexNetSiLU.py')
            paths['model_source']=paths['baseline_root']/'frontend/dacapo/python/poly/poly/models'/source_names[suffix]
            paths['model_source'].parent.mkdir(parents=True,exist_ok=True)
            paths['model_source'].write_text('# fixture source is replaced by a tiny model builder\n')
        args=SimpleNamespace(baseline=baseline,model=model,**paths)
        cli=sum((['--'+key.replace('_','-'),str(value)] for key,value in vars(args).items()),[])
        return SimpleNamespace(args=args,cli=cli,info=info,graph=graph,original=original,
                               source=source,references=references,images=images,factories=factories)


class OrionCases(_SmallCases, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from pt_agent.experiments import r16_orion_models as adapter
        cls.adapter=adapter

    def test_complete_eleven_constants_have_explicit_roles_and_float64_values(self):
        torch=self.torch
        expected={
            'resnet20_silu':dict(**{'module.layer2.0.shortcut.weight':(32,16,1,1),
                                    'module.layer3.0.shortcut.weight':(64,32,1,1),
                                    'module._r16_avgpool.weight':(1,1,8,8)}),
            'vgg16_silu_avgpool':{'avgpool_'+str(i)+'.weight':(1,1,2,2) for i in range(1,6)},
            'alexnet_silu_avgpool':{'avgpool_'+str(i)+'.weight':(1,1,3,3) for i in range(1,4)}}
        all_specs=[]
        for suffix,shapes in expected.items():
            with self.subTest(model=suffix):
                model='pt_agent.experiments.r13_common_models:'+suffix
                specs=self.adapter._extra_spec(model)
                values=self.adapter._extra_state(model)
                self.assertEqual({entry['name']:tuple(entry['shape']) for entry in specs},shapes)
                self.assertEqual(set(values),set(shapes))
                for entry in specs:
                    name=entry['name'];value=values[name]
                    self.assertEqual(tuple(value.shape),shapes[name])
                    self.assertEqual(value.dtype,torch.float64)
                    self.assertFalse(value.requires_grad)
                    self.assertEqual(value.device.type,'cpu')
                    if 'shortcut' in name:
                        self.assertFalse(entry['unused_by_plain_forward'])
                    else:
                        self.assertTrue(entry['unused_by_plain_forward'])
                        kernel=shapes[name][-1]
                        torch.testing.assert_close(value,torch.full(shapes[name],1/(kernel*kernel),
                            dtype=torch.float64),rtol=0,atol=0)
                all_specs.extend(specs)
        self.assertEqual(len(all_specs),11)
        self.assertEqual(sum(s['unused_by_plain_forward'] for s in all_specs),9)
        self.assertEqual(len({s['role'] for s in all_specs}),2)

    def test_option_a_zero_one_selection_matches_padding_on_even_and_odd_spatial_sizes(self):
        torch=self.torch
        import torch.nn.functional as functional

        for channels in (16,32):
            weight=self.adapter._option_a_weight(channels)
            expected=torch.zeros((2*channels,channels,1,1),dtype=torch.float64)
            for c in range(channels):expected[channels//2+c,c,0,0]=1.
            torch.testing.assert_close(weight,expected,rtol=0,atol=0)
            self.assertEqual(int((weight==1).sum()),channels)
            self.assertEqual(int(((weight==0)|(weight==1)).sum()),weight.numel())
            for height,width in ((8,8),(7,9),(9,6)):
                with self.subTest(channels=channels,shape=(height,width)):
                    image=torch.arange(channels*height*width,dtype=torch.float64).reshape(1,channels,height,width)
                    selected=functional.conv2d(image,weight,stride=2)
                    sliced=image[:,:,::2,::2]
                    padded=functional.pad(sliced,(0,0,0,0,channels//2,channels//2))
                    torch.testing.assert_close(selected,padded,rtol=0,atol=0)

    def test_corrupt_selector_or_float32_rounded_pool_constant_is_rejected(self):
        for model in (self.adapter.RESNET,self.adapter.ALEX):
            with self.subTest(model=model):
                extras=self.adapter._extra_state(model)
                self.adapter._verify_extra_values(model,extras)
                name=next(iter(extras))
                if model==self.adapter.RESNET:extras[name][0,0,0,0]=0.5
                else:extras[name]=extras[name].float().double()
                with self.assertRaises(ValueError):self.adapter._verify_extra_values(model,extras)

    def test_pool_weight_is_explicitly_unused_by_plain_forward_and_eval_returns_none(self):
        torch,on,scheme=self.adapter._runtime()
        pool=on.AvgPool2d(3,stride=2,padding=1).cpu().double()
        self.assertIsNone(pool.eval())
        self.assertFalse(pool.he_mode)
        image=torch.arange(2*7*9,dtype=torch.float64).reshape(1,2,7,9)/10
        with torch.no_grad():
            pool.weight.fill_(0.)
            first=pool(image)
            pool.weight.fill_(123.)
            second=pool(image)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertIsNotNone(scheme)
        self.assertIsNone(scheme.backend)

    def _tiny_orion_state(self):
        torch,on,_=self.adapter._runtime()
        class Tiny(on.Module):
            def __init__(self):
                super().__init__()
                self.linear=torch.nn.Linear(2,2)
                self.bn=torch.nn.BatchNorm1d(2)
                self.pool=on.AvgPool2d(3,stride=2,padding=1)
            def forward(self,value):
                return self.bn(self.linear(value))
        module=Tiny().cpu().double()
        state={name:(value.detach().to(torch.float32).clone() if value.is_floating_point()
                     else value.detach().clone()) for name,value in module.state_dict().items()
               if name!='pool.weight'}
        state['bn.num_batches_tracked']=torch.tensor(7,dtype=torch.int64)
        for name,value in state.items():
            if value.is_floating_point():value.fill_(0.1 if 'running_var' not in name else 1.1)
        extra={'pool.weight':torch.full((1,1,3,3),1/9,dtype=torch.float64)}
        return module,state,extra

    def test_strict_state_preserves_original_values_bn_integers_and_independent_extra_copy(self):
        module,state,extras=self._tiny_orion_state()
        before={name:value.clone() for name,value in state.items()}
        report=self.adapter._load_state(module,state,extras)
        actual=module.state_dict()
        self.assertEqual(set(actual)-set(extras),set(state))
        for name,value in state.items():
            target=value.double() if value.is_floating_point() else value
            self.torch.testing.assert_close(actual[name],target,rtol=0,atol=0)
            self.torch.testing.assert_close(state[name],before[name],rtol=0,atol=0)
        self.assertEqual(actual['bn.num_batches_tracked'].dtype,self.torch.int64)
        self.torch.testing.assert_close(actual['pool.weight'],extras['pool.weight'],rtol=0,atol=0)
        self.assertFalse(module.pool.weight.requires_grad)
        self.assertTrue(report['original_keys_preserved'])
        self.assertTrue(report['strict_load'])

    def test_strict_state_rejects_missing_unexpected_and_wrong_shape_keys(self):
        for case in ('missing','unexpected','shape'):
            with self.subTest(case=case):
                module,state,extras=self._tiny_orion_state()
                if case=='missing':state.pop('bn.running_mean')
                elif case=='unexpected':state['unexpected']=self.torch.zeros(1)
                else:state['linear.weight']=self.torch.zeros((3,2))
                with self.assertRaises((ValueError,RuntimeError)):
                    self.adapter._load_state(module,state,extras)

    def test_build_converts_to_double_before_creating_one_ninth_constants_and_keeps_eval_model(self):
        torch,on,scheme=self.adapter._runtime()
        events=[]
        class Tiny(on.Module):
            def __init__(self):
                super().__init__()
                self.linear=torch.nn.Linear(2,2)
                self.bn=torch.nn.BatchNorm1d(2)
                for i in range(1,4):setattr(self,'avgpool_'+str(i),on.AvgPool2d(3,stride=2,padding=1))
            def cpu(self):events.append('cpu');return super().cpu()
            def double(self):events.append('double');return super().double()
            def eval(self):events.append('eval');super().eval()
            def forward(self,value):return self.bn(self.linear(value))
        tiny=Tiny()
        original={name:value.detach().clone() for name,value in tiny.state_dict().items()
                  if not name.startswith('avgpool_')}
        make_extras=self.adapter._extra_state
        load_state=self.adapter._load_state
        def extras(model):
            events.append('extras')
            self.assertTrue(all(value.dtype==torch.float64 for value in tiny.parameters()))
            return make_extras(model)
        def load(module,state,extra):
            events.append('strict_load')
            return load_state(module,state,extra)
        with patch.object(self.adapter,'_construct_model',return_value=tiny), \
             patch.object(self.adapter,'_extra_state',side_effect=extras), \
             patch.object(self.adapter,'_load_state',side_effect=load):
            model,report=self.adapter.build_model(self.adapter.ALEX,original)
        self.assertIs(model,tiny)
        self.assertEqual(events,['cpu','double','extras','strict_load','eval'])
        self.assertTrue(report['passed'])
        self.assertTrue(report['structure']['passed'])
        self.assertTrue(report['all_he_mode_false'])
        self.assertTrue(report['backend_is_none'])
        self.assertIsNone(scheme.backend)
        self.assertFalse(report['packing_verified'])
        self.assertFalse(report['fhe_activation_scheme_frozen'])
        for key in ('init_scheme_calls','backend_initializations','keygen_runs','fit_calls','compile_runs'):
            self.assertEqual(report[key],0)

    def test_original_class_call_order_and_heads_are_inspected_without_large_construction(self):
        tree=ast.parse(inspect.getsource(self.adapter._model_classes))
        classes={node.name:node for node in ast.walk(tree) if isinstance(node,ast.ClassDef)}
        def method(name,which):
            return next(n for n in classes[name].body if isinstance(n,ast.FunctionDef) and n.name==which)
        def calls_in_order(node):
            order=[]
            def visit(value):
                if isinstance(value,ast.Call):
                    for argument in value.args:visit(argument)
                    if isinstance(value.func,ast.Attribute) and isinstance(value.func.value,ast.Name) and value.func.value.id=='self':
                        order.append(value.func.attr)
            for statement in node.body:
                if isinstance(statement,(ast.Assign,ast.Expr)):visit(statement.value)
            return order
        self.assertEqual(calls_in_order(method('BasicConv2d','forward')),['Conv2d','bn','mish'])
        self.assertEqual(calls_in_order(method('BasicBlock','forward')),
                         ['conv1','bn1','mish','conv2','bn2','shortcut','_r16_add','mish'])
        self.assertEqual(calls_in_order(method('VGG16','forward'))[-7:],
                         ['_r16_flatten','fc_1','mish','fc_2','mish','bn_1','fc_3'])
        self.assertEqual(calls_in_order(method('AlexNet','forward'))[-6:],
                         ['_r16_flatten','fc_1','mish','fc_2','mish','fc_3'])
        for name in ('VGG16','AlexNet'):
            self.assertFalse(any(item.startswith('dp_') for item in calls_in_order(method(name,'forward'))))
        def integer(expression):
            if isinstance(expression,ast.Constant):return expression.value
            self.assertIsInstance(expression,ast.BinOp)
            self.assertIsInstance(expression.op,ast.Mult)
            return integer(expression.left)*integer(expression.right)
        def constructor_arguments(name,constructor):
            return [[integer(arg) for arg in n.value.args] for n in method(name,'__init__').body
                    if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
                    and isinstance(n.value.func,ast.Attribute) and n.value.func.attr==constructor]
        self.assertEqual(constructor_arguments('VGG16','Linear'),[[512,256],[256,128],[128,10]])
        self.assertEqual(constructor_arguments('AlexNet','Linear'),[[4096,2048],[2048,1024],[1024,10]])
        self.assertEqual(constructor_arguments('AlexNet','Conv2d'),
                         [[3,96,3],[96,256,5],[256,384,3],[384,384,3],[384,256,3]])


class OrbitCases(_SmallCases, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from pt_agent.experiments import r16_orbit_models as adapter
        cls.adapter=adapter

    def _definition(self,model):
        record=self.adapter.MODEL_RECORDS[model]
        main={'resnet20_silu':'ResNetSiLU','vgg16_silu_avgpool':'VGG16SiLU',
              'alexnet_silu_avgpool':'AlexNetSiLU'}[model.rsplit(':',1)[-1]]
        pieces=[]
        for name in record['definitions']:
            if name==record['factory']:
                pieces.append('def '+name+'():\n    return '+main+'()\n')
            elif name=='_weights_init':pieces.append('def _weights_init(value):\n    return None\n')
            else:
                pieces.append('class '+name+'(nn.Module):\n'
                    '    def __init__(self):\n'
                    '        super().__init__()\n'
                    '        self.used_cuda = use_cuda\n'
                    '        self.linear = nn.Linear(2,2)\n'
                    '    def forward(self,value):\n'
                    '        return self.linear(value)\n')
        return '\n'.join(pieces)

    def test_ast_loads_only_the_three_exact_definition_sets_and_forces_cpu(self):
        for index,(model,record) in enumerate(self.adapter.MODEL_RECORDS.items()):
            with self.subTest(model=model):
                sentinel=self.root/('side-effect-'+str(index))
                source=self.root/record['source_name']
                source.write_text('import missing_top_level_package\nuse_cuda=True\n'
                    'open('+repr(str(sentinel))+',"w").write("must not execute")\n'
                    'raise RuntimeError("top-level execution")\n'+self._definition(model)+'\n'
                    'if __name__ == "__main__":\n    raise RuntimeError("main execution")\n')
                loaded=self.adapter.load_definition(source,model)
                self.assertFalse(sentinel.exists())
                self.assertFalse(loaded.use_cuda)
                self.assertTrue(all(hasattr(loaded,name) for name in record['definitions']))
                self.assertIs(loaded.torch,self.torch)
                self.assertIs(loaded.nn,self.torch.nn)
                tiny=getattr(loaded,record['factory'])()
                self.assertFalse(tiny.used_cuda)
                self.assertTrue(all(value.device.type=='cpu' for value in tiny.parameters()))

    def test_ast_rejects_missing_duplicate_async_and_decorated_definitions(self):
        model=next(iter(self.adapter.MODEL_RECORDS))
        factory=self.adapter.MODEL_RECORDS[model]['factory']
        for case in ('missing','duplicate','async','decorator'):
            with self.subTest(case=case):
                text=self._definition(model)
                if case=='missing':text=text.replace('def '+factory+'():','def another_factory():')
                elif case=='duplicate':text+='\ndef '+factory+'():\n    return None\n'
                elif case=='async':text=text.replace('def '+factory+'():','async def '+factory+'():')
                else:text=text.replace('def '+factory+'():','@forbidden_decorator\ndef '+factory+'():')
                source=self.root/(case+'.py');source.write_text(text)
                with self.assertRaises(ValueError):self.adapter.load_definition(source,model)

    def _tiny_state(self):
        torch=self.torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear=torch.nn.Linear(2,2)
                self.bn=torch.nn.BatchNorm1d(2)
            def forward(self,value):return self.bn(self.linear(value))
        module=Tiny().cpu().double()
        state={name:(value.float().clone() if value.is_floating_point() else value.clone())
               for name,value in module.state_dict().items()}
        state['bn.num_batches_tracked']=torch.tensor(11,dtype=torch.int64)
        return module,state

    def test_strict_load_preserves_bn_dtype_float_values_and_original_dictionary(self):
        module,state=self._tiny_state()
        before={name:value.clone() for name,value in state.items()}
        rows=self.adapter._strict_load(module,state)
        self.assertEqual({row['name'] for row in rows},set(state))
        for row in rows:
            name=row['name'];source=state[name]
            expected=source.double() if source.is_floating_point() else source
            self.torch.testing.assert_close(module.state_dict()[name],expected,rtol=0,atol=0)
            self.torch.testing.assert_close(state[name],before[name],rtol=0,atol=0)
            self.assertTrue(row['finite'])
            self.assertTrue(row['numerically_equal'])
        self.assertEqual(module.bn.num_batches_tracked.dtype,self.torch.int64)

    def test_strict_load_rejects_keys_shapes_nonfinite_and_preconversion_state(self):
        for case in ('missing','unexpected','shape','nonfinite','not_double'):
            with self.subTest(case=case):
                module,state=self._tiny_state()
                if case=='missing':state.pop('bn.running_mean')
                elif case=='unexpected':state['extra']=self.torch.ones(1)
                elif case=='shape':state['linear.weight']=self.torch.ones((3,2))
                elif case=='nonfinite':state['linear.weight'][0,0]=self.torch.nan
                else:module.float()
                with self.assertRaises(ValueError):self.adapter._strict_load(module,state)

    def test_build_converts_before_strict_load_and_keeps_model_when_eval_returns_none(self):
        torch=self.torch
        events=[]
        class Tiny(torch.nn.Module):
            def __init__(self):super().__init__();self.linear=torch.nn.Linear(2,2)
            def cpu(self):events.append('cpu');return super().cpu()
            def double(self):events.append('double');return super().double()
            def eval(self):events.append('eval');super().eval()
        tiny=Tiny();state={name:value.clone() for name,value in tiny.state_dict().items()}
        model='pt_agent.experiments.r13_common_models:vgg16_silu_avgpool'
        definition=SimpleNamespace(vgg16silu=lambda:tiny)
        original=self.adapter._strict_load
        def strict(module,values):events.append('strict');return original(module,values)
        with patch.object(self.adapter,'load_definition',return_value=definition), \
             patch.object(self.adapter,'_strict_load',side_effect=strict), \
             patch.object(self.adapter,'_validate_structure',return_value=dict(passed=True,errors=[])):
            actual,report=self.adapter.build_model(model,self.root/'source.py',state)
        self.assertIs(actual,tiny)
        self.assertEqual(events,['cpu','double','strict','eval'])
        self.assertTrue(report['passed'])
        self.assertEqual(report['extra_state'],[])

    def test_original_source_interface_policy_and_preprocessing_are_preserved(self):
        for suffix in ('resnet20_silu','vgg16_silu_avgpool','alexnet_silu_avgpool'):
            fixture=self._fixture(suffix=suffix)
            before=copy.deepcopy((fixture.original,fixture.info,fixture.graph))
            with patch.object(self.worker,'FACTORIES',fixture.factories):
                checked=self.worker.validate_source(fixture.original,fixture.info,fixture.graph)
            self.assertTrue(checked['passed'],checked)
            self.assertFalse(checked['training_preprocessing_verified'])
            self.assertEqual((fixture.original,fixture.info,fixture.graph),before)

    def test_original_source_rejects_features_task_policy_split_and_reference_changes(self):
        fixture=self._fixture()
        for case in ('features','task','policy','split','reference','shape','preprocessing'):
            with self.subTest(case=case):
                r13,info,graph=copy.deepcopy((fixture.original,fixture.info,fixture.graph))
                if case=='features':info['inputs'][0]['kind']='image'
                elif case=='task':info['task']='regression'
                elif case=='policy':info['threshold']={'acc_drop_pp':1.}
                elif case=='split':info['split']['report']=[]
                elif case=='reference':r13['worker']['graph_math']['passed']=False
                elif case=='shape':graph['tensors'][-1][1]=[1,11]
                else:info['source']['training_preprocessing_verified']=True
                with patch.object(self.worker,'FACTORIES',fixture.factories):
                    self.assertFalse(self.worker.validate_source(r13,info,graph)['passed'])

    def test_math_preserves_the_exact_formula_and_inclusive_zero_reference_boundary(self):
        reference=self.np.zeros((1,10),dtype=self.np.float64)
        observed=reference.copy();observed[0,0]=1e-12
        checked=self.worker.check_math(reference,observed)
        self.assertTrue(checked['passed'])
        self.assertEqual(checked['criterion'],'1e-12+1e-12*abs(reference)')
        observed[0,0]=self.np.nextafter(1e-12,self.np.inf)
        self.assertFalse(self.worker.check_math(reference,observed)['passed'])
        reference[0,1]=1e6;observed=reference.copy()
        observed[0,1]+=(1e-12+1e-12*abs(reference[0,1]))*0.5
        self.assertTrue(self.worker.check_math(reference,observed)['passed'])

    def test_math_rejects_shape_original_dtype_and_nonfinite_before_conversion(self):
        valid=self.np.zeros((1,10),dtype=self.np.float64)
        for observed in (valid.ravel(),valid.astype(self.np.float32),
                         self.np.full((1,10),self.np.nan),self.np.full((1,10),self.np.inf)):
            self.assertFalse(self.worker.check_math(valid,observed)['passed'])
        self.assertFalse(self.worker.check_math(valid.astype(self.np.float32),valid)['passed'])

    def _invoke(self,fixture,*,failure=None,index=1):
        from pt_agent.data import cifar
        from pt_agent.intake.graph import Graph
        from pt_agent.models.info import ModelInfo

        torch=self.torch
        calls=[]
        class TinyProbe(torch.nn.Module):
            def forward(module,image):
                self.assertFalse(torch.is_grad_enabled())
                self.assertEqual(image.dtype,torch.float64)
                self.assertEqual(tuple(image.shape),(1,3,32,32))
                i=int(image[0,0,0,0]);calls.append(i)
                if failure=='runtime' and i==index:raise RuntimeError('fixture forward failure')
                value=torch.arange(10,dtype=torch.float64).reshape(1,10)+i
                if i==index:
                    if failure=='math':value[0,0]+=1e-6
                    elif failure=='dtype':value=value.float()
                    elif failure=='shape':value=value.reshape(10)
                    elif failure=='nonfinite':value[0,0]=torch.nan
                return value
        state={'tiny.weight':torch.tensor([0.1,0.2],dtype=torch.float32)}
        checkpoint={'state_dict':state,'epoch':7} if fixture.args.model.endswith(':resnet20_silu') else state
        report=dict(passed=True,strict_load=True,structure=dict(passed=True,errors=[]),
                    original_state=[dict(name='tiny.weight',shape=[2],original_dtype='torch.float32',
                                         target_dtype='torch.float64',finite=True,numerically_equal=True)],
                    extra_state=[])
        original_load=self.np.load
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.worker,'FACTORIES',fixture.factories))
            stack.enter_context(patch.object(self.worker,'_initialize_torch'))
            load_checkpoint=stack.enter_context(patch.object(torch,'load',return_value=checkpoint))
            load_probes=stack.enter_context(patch.object(self.np,'load',wraps=original_load))
            build=stack.enter_context(patch.object(self.worker,'_build_model',return_value=(TinyProbe(),report)))
            denied=[stack.enter_context(patch.object(target,name,side_effect=AssertionError('forbidden '+name)))
                    for target,name in ((ModelInfo,'load'),(Graph,'load'),(cifar,'read_rows'))]
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            code=self.worker.main(fixture.cli)
            for action in denied:action.assert_not_called()
        result=json.loads((fixture.args.work_dir/'worker_result.json').read_text())
        return code,result,calls,load_checkpoint,load_probes,build

    def _assert_unmeasured(self,result):
        self.assertFalse(result['candidate_frozen'])
        for field in self.worker.ZERO_FIELDS:self.assertEqual(result[field],0)
        for field in self.worker.UNMEASURED_FIELDS:self.assertIsNone(result[field])

    def test_worker_reads_checkpoint_and_saved_probes_once_and_forwards_in_original_order(self):
        fixture=self._fixture()
        code,result,calls,checkpoint,probes,build=self._invoke(fixture)
        self.assertEqual(code,0,result.get('error'))
        self.assertEqual(result['status'],'ok')
        self.assertTrue(result['compatibility_passed'])
        self.assertEqual(calls,[0,1,2])
        checkpoint.assert_called_once_with(fixture.args.checkpoint,map_location='cpu',weights_only=True)
        probes.assert_called_once_with(fixture.args.probes,allow_pickle=False)
        build.assert_called_once()
        for key in self.worker.COUNT_FIELDS:self.assertEqual(result[key],3 if key.startswith('forward_') else 1)
        for key in self.worker.STAGE_FIELDS:self.assertTrue(result[key])
        self.assertEqual([p['sample_id'] for p in result['math_probes']],['train:0','train:1','train:2'])
        self.assertEqual(len(result['math_probes']),3)
        self._assert_unmeasured(result)

    def test_resnet_checkpoint_envelope_and_module_keys_are_preserved_by_one_read(self):
        fixture=self._fixture(suffix='resnet20_silu')
        fixture.args.work_dir.mkdir(parents=True)
        state={'module.bn.num_batches_tracked':self.torch.tensor(3,dtype=self.torch.int64),
               'module.weight':self.torch.tensor([0.1],dtype=self.torch.float32)}
        checkpoint=dict(state_dict=state,epoch=7)
        before={name:value.clone() for name,value in state.items()}
        result=self.worker._initial_result()
        with patch.object(self.torch,'load',return_value=checkpoint) as load:
            actual=self.worker._checkpoint(fixture.args,result)
        self.assertIs(actual,state)
        self.assertEqual(list(checkpoint),['state_dict','epoch'])
        for name in state:self.torch.testing.assert_close(state[name],before[name],rtol=0,atol=0)
        self.assertEqual(result['checkpoint_load_attempts'],1)
        self.assertEqual(result['checkpoint_load_completed'],1)
        self.assertFalse(result['checkpoint_envelope']['original_dictionary_modified'])
        load.assert_called_once_with(fixture.args.checkpoint,map_location='cpu',weights_only=True)

    def test_worker_math_failure_stops_at_the_actual_probe_and_keeps_partial_counts(self):
        fixture=self._fixture()
        code,result,calls,_,_,_=self._invoke(fixture,failure='math')
        self.assertEqual(code,1)
        self.assertEqual(result['status'],'math_failed')
        self.assertEqual(calls,[0,1])
        self.assertEqual(result['forward_attempts'],2)
        self.assertEqual(result['forward_completed'],2)
        self.assertFalse(result['math_probes_passed'])
        self.assertIsNone(result['compatibility_passed'])
        self._assert_unmeasured(result)

    def test_worker_runtime_failure_preserves_attempted_and_completed_separately(self):
        fixture=self._fixture()
        code,result,calls,_,_,_=self._invoke(fixture,failure='runtime')
        self.assertEqual(code,1)
        self.assertEqual(result['status'],'error')
        self.assertEqual(calls,[0,1])
        self.assertEqual(result['forward_attempts'],2)
        self.assertEqual(result['forward_completed'],1)
        self.assertIsNone(result['math_probes_passed'])
        self.assertIsNone(result['compatibility_passed'])
        self._assert_unmeasured(result)

    def test_worker_original_dtype_shape_and_finite_checks_precede_numpy_conversion(self):
        for failure,status in (('dtype','shape_failed'),('shape','shape_failed'),('nonfinite','nonfinite')):
            with self.subTest(failure=failure):
                fixture=self._fixture()
                code,result,calls,_,_,_=self._invoke(fixture,failure=failure)
                self.assertEqual(code,1)
                self.assertEqual(result['status'],status)
                self.assertEqual(calls,[0,1])
                self.assertEqual(result['forward_completed'],2)
                self.assertEqual(len(result['math_probes']),1)
                self._assert_unmeasured(result)
                self.root=self.root/('after-'+failure);self.root.mkdir()

    def test_invalid_source_stops_before_checkpoint_probe_reads_or_model_build(self):
        fixture=self._fixture()
        fixture.info['threshold']['acc_drop_pp']=1.
        self._json(fixture.args.model_info,fixture.info)
        code,result,calls,checkpoint,probes,build=self._invoke(fixture)
        self.assertEqual(code,1)
        self.assertEqual(result['status'],'source_failed')
        self.assertEqual(calls,[])
        checkpoint.assert_not_called();probes.assert_not_called();build.assert_not_called()
        for key in self.worker.COUNT_FIELDS:self.assertEqual(result[key],0)
        self.assertIsNone(result['checkpoint_checked'])
        self.assertIsNone(result['compatibility_passed'])
        self._assert_unmeasured(result)

    def test_probe_source_and_npz_indices_match_saved_original_order(self):
        fixture=self._fixture()
        fixture.args.work_dir.mkdir(parents=True)
        for case in ('source','index_order','dtype','extra_field','nonfinite','reference_order'):
            with self.subTest(case=case):
                source=copy.deepcopy(fixture.source)
                r13=copy.deepcopy(fixture.original)
                images=fixture.images.copy()
                values=dict(input=images,labels=self.np.array(source['labels']),indices=self.np.array(source['indices']))
                if case=='source':source['split']='heldout'
                elif case=='index_order':values['indices']=values['indices'][::-1]
                elif case=='dtype':values['input']=images.astype(self.np.float32)
                elif case=='extra_field':values['unapproved']=self.np.array([1.])
                elif case=='nonfinite':values['input'][0,0,0,0]=self.np.nan
                else:r13['worker']['graph_math']['samples'][0]['probe_index']=2
                self._json(fixture.args.probe_source,source)
                self.np.savez(fixture.args.probes,**values)
                result=self.worker._initial_result()
                with self.assertRaises(self.worker.WorkerFailure):
                    self.worker._read_probes(fixture.args,r13,fixture.info,fixture.graph,result)
