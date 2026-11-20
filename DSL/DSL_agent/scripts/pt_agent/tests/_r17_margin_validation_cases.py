"""R17 inner checks use synthetic graphs, cached metadata and numeric fakes."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class MarginCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name]='1'
        from pt_agent.experiments import r17_margin_validation_worker as worker
        worker._initialize_torch()
        import numpy as np
        import torch
        from pt_agent.data import cifar
        from pt_agent.dsl.parser import parse
        from pt_agent.experiments.r13_common_models import FACTORIES
        from pt_agent.intake.graph import Graph, Node, TensorInfo
        from pt_agent.models.info import ModelInfo
        from pt_agent.planners import rule

        cls.np,cls.torch,cls.worker=np,torch,worker
        cls.cifar,cls.factories=cifar,FACTORIES
        cls.Graph,cls.Node,cls.TensorInfo,cls.ModelInfo=Graph,Node,TensorInfo,ModelInfo
        cls.parse,cls.rule=staticmethod(parse),rule

    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='r17-margin-cases-')
        self.addCleanup(temporary.cleanup)
        self.directory=Path(temporary.name)

    def write_json(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def fixture(self,suffix='resnet20_silu',*,act_count=None):
        np=self.np
        model='pt_agent.experiments.r13_common_models:'+suffix
        original=self.factories[model]
        expected_acts={'resnet20_silu':19,'vgg16_silu_avgpool':15,'alexnet_silu_avgpool':7}[suffix]
        count=expected_acts if act_count is None else act_count
        shapes={'camera_features':(1,3,32,32),'stem output':(1,16,32,32),
                'pool.value[0]':(1,16),'declared_logits':(1,10),
                'stem.weight':(16,3,1,1),'head.weight':(10,16),'head.bias':(10,)}
        nodes=[self.Node('conv','conv2d',['camera_features'],['stem output'],
                         dict(weight='stem.weight',stride=(1,1),padding=(0,0),dilation=(1,1),groups=1),'fixture conv'),
               self.Node('pool','mean',['stem output'],['pool.value[0]'],
                         dict(axes=(2,3),keepdim=False),'fixture mean')]
        previous='pool.value[0]';calibration={}
        for index in range(count):
            output='internal:layer.'+str(index)+'/value'
            shapes[output]=(1,16)
            nodes.append(self.Node('graph-act-'+str(index),'act',[previous],[output],dict(fn='silu'),'fixture SiLU'))
            low,high=(-2.,2.) if index%2==0 else (-3.,1.)
            calibration[previous]=dict(min=low,max=high,absmax=max(abs(low),abs(high)),p999=high,count=16000)
            previous=output
        nodes.append(self.Node('head','linear',[previous],['declared_logits'],
                               dict(weight='head.weight',bias='head.bias'),'fixture logits'))
        tensors={name:self.TensorInfo(name,shape,'input' if name=='camera_features' else
                                     'weight' if name.endswith(('weight','bias')) else 'act')
                 for name,shape in shapes.items()}
        weights={'stem.weight':np.full((16,3,1,1),0.01,dtype=np.float64),
                 'head.weight':np.arange(160,dtype=np.float64).reshape(10,16)/100,
                 'head.bias':np.arange(10,dtype=np.float64)/1000}
        graph=self.Graph('r17_small_fixture',nodes,tensors,weights,['camera_features'],['declared_logits'])
        rows={'resnet20_silu':74,'vgg16_silu_avgpool':50,'alexnet_silu_avgpool':24}[suffix]
        for index in range(rows-len(calibration)):
            calibration['cached.observation.'+str(index)]=dict(min=-1.,max=1.,absmax=1.,p999=1.,count=32000)
        norm=dict(mean=list(self.cifar.MEAN),std=list(self.cifar.STD))
        source=dict(type='torch',path=model,factory=model,spec_revision='R13',
                    original_source=str(original['source'].resolve()),checkpoint=str(original['checkpoint'].resolve()),
                    inference_preprocessing_source=str(original['inference_preprocessing_source'].resolve()),
                    normalization=norm,calibration_source=dict(split='train',indices=[0,1,2]),
                    preprocessing_provenance='official inference test',training_preprocessing_verified=False,
                    intake=dict(original_nodes=len(nodes),folded_nodes=0,original_unmapped={}))
        split=dict(kind='legacy_cifar',heldout=list(self.cifar.HELDOUT),confirm=list(self.cifar.CONFIRM),
                   report=list(self.cifar.REPORT),calibration='frozen training selection')
        info=self.ModelInfo(source,graph,dict(internal_regression_source='legacy_cifar'),
            [dict(name='camera_features',kind='features',shape=[1,3,32,32],dtype='float64')],
            None,'classification','classification_top1',dict(acc_drop_pp=0.5),split,[])
        root=self.directory/suffix
        r13_dir=root/('r13-common-'+suffix.replace('_','-')+'-20261115')
        r14_dir=root/('r14-common-'+suffix.replace('_','-')+'-20261115')
        model_info=r13_dir/'worker/model_info';info.save(model_info)
        r13=dict(round='R13',mode='common-model-intake-preparation',status='ok',model=model,
            worker=dict(round='R13',status='ok',model=model,model_source=copy.deepcopy(source),
                        split=copy.deepcopy(split),import_completed=True,real_probes_passed=True,
                        graph_math_passed=True,roundtrip_passed=True,static_coverage=True,
                        artifacts=dict(model_info=str(model_info))))
        r13_path=r13_dir/'result.json';self.write_json(r13_path,r13)
        work=r14_dir/'worker';work.mkdir(parents=True)
        calibration_path=work/'calibration.json';cal_source_path=work/'calibration_source.json'
        original_program=work/'program.ptdsl'
        options=dict(deg=63,margin=1.25,slots=16384,placement='depth-dp',refresh='compiler',act_method='cheb')
        original_text=self.rule.plan(graph,calibration,**options)
        original_program.write_text(original_text)
        calibration_source=dict(records=str(self.directory/'raw-training-records'),
            selections=str(self.directory/'frozen-selections.json'),split='train',count=1000,batch=1,
            indices=list(range(1000)),sample_ids=['train:'+str(i) for i in range(1000)],
            entries=[dict(index=i,label=i%10,shard='train-00000.records',row=i) for i in range(1000)],
            normalization=norm,dataset_metadata=dict(format='poseidon-cifar10-binary-records-v1',record_bytes=3073,
              row_label_byte=0,preprocessing_applied=False,train_count=50000,test_count=10000,
              pixel_layout='uint8 planar RGB 3x32x32, C order'))
        saved_stats=dict(counts=dict(add=2,sub=0,neg=0,mul_cc=1,mul_cp=1,rot=0,boot=0),
                         ops_total=4,distinct_consts=2)
        worker=dict(round='R14',model=model,status='ok' if suffix=='resnet20_silu' else 'artifact_budget_exceeded',
            failed_phase=None if suffix=='resnet20_silu' else 'emit_write',
            r13_source_result=str(r13_path),rule_options=options,
            source_checked=True,reference_passed=True,calibration_completed=True,plan_completed=True,
            typecheck_passed=True,signature_passed=True,count_completed=True,stats_equal=True,
            emit_completed=True if suffix=='resnet20_silu' else None,
            lowering_ready=True if suffix=='resnet20_silu' else None,
            calibration_source=copy.deepcopy(calibration_source),
            calibration_summary=dict(count=1000,batch=1,fields=list(next(iter(calibration.values()))),
                p999='maximum of each batch 99.9th absolute percentile; batch=1'),
            count_stats=copy.deepcopy(saved_stats),
            artifacts=dict(calibration=str(calibration_path),calibration_source=str(cal_source_path),
                           program=str(original_program)))
        r14=dict(round='R14',mode='common-model-rule-preparation',status=worker['status'],model=model,worker=worker,
            source_r13=dict(result=str(r13_path),run=str(r13_dir),artifacts=dict(model_info=str(model_info))))
        r14_path=r14_dir/'result.json';self.write_json(r14_path,r14)
        self.write_json(calibration_path,calibration);self.write_json(cal_source_path,calibration_source)
        records=root/'heldout-records';records.mkdir()
        paths=dict(r14_result=r14_path,r13_result=r13_path,model_info=model_info,
                   calibration=calibration_path,calibration_source=cal_source_path,
                   original_program=original_program,records=records,work_dir=root/'new-validation/worker')
        return SimpleNamespace(model=model,suffix=suffix,info=info,r13=r13,r14=r14,calibration=calibration,
            calibration_source=calibration_source,original_text=original_text,saved_stats=saved_stats,
            args=SimpleNamespace(**paths),cli=sum((['--'+k.replace('_','-'),str(v)] for k,v in paths.items()),[]))

    def test_all_three_cached_sources_preserve_failed_emit_and_original_policy(self):
        for suffix in ('resnet20_silu','vgg16_silu_avgpool','alexnet_silu_avgpool'):
            with self.subTest(model=suffix):
                f=self.fixture(suffix)
                before=copy.deepcopy((f.info.source,f.info.inputs,f.info.split,f.info.task,f.info.threshold))
                checked=self.worker.validate_source(f.r14,f.r13,f.info,f.calibration,f.calibration_source)
                self.assertTrue(checked['passed'],checked)
                self.assertEqual((f.info.source,f.info.inputs,f.info.split,f.info.task,f.info.threshold),before)
                self.assertEqual(f.r14['status'],'ok' if suffix=='resnet20_silu' else 'artifact_budget_exceeded')
                self.assertEqual(len(f.calibration),{'resnet20_silu':74,'vgg16_silu_avgpool':50,'alexnet_silu_avgpool':24}[suffix])
                self.assertNotIn('labels',f.calibration_source)

    def test_cache_source_is_compared_with_independently_saved_observation(self):
        f=self.fixture()
        changed=copy.deepcopy(f.calibration_source)
        changed['entries'][1]['label']=8
        self.assertFalse(self.worker.validate_source(f.r14,f.r13,f.info,f.calibration,changed)['passed'])
        self.assertNotEqual(changed,f.r14['worker']['calibration_source'])

    def test_failed_reference_phase_wrong_emit_phase_or_false_success_is_rejected(self):
        f=self.fixture('vgg16_silu_avgpool')
        for case in ('reference','calibration','count','wrong_failure','false_success'):
            with self.subTest(case=case):
                r14=copy.deepcopy(f.r14)
                if case=='reference':r14['worker']['reference_passed']=False
                elif case=='calibration':r14['worker']['calibration_completed']=False
                elif case=='count':r14['worker']['count_completed']=False
                elif case=='wrong_failure':r14['worker']['failed_phase']='calibrate'
                else:r14['status']=r14['worker']['status']='ok'
                self.assertFalse(self.worker.validate_source(r14,f.r13,f.info,f.calibration,f.calibration_source)['passed'])

    def test_cache_shape_finite_count_origin_policy_and_source_identities_are_checked(self):
        f=self.fixture()
        for case in ('missing_point','finite','scalar_count','observations','duplicate','label','shard','norm','policy','features'):
            with self.subTest(case=case):
                r14,r13,info,cal,source=copy.deepcopy((f.r14,f.r13,f.info,f.calibration,f.calibration_source))
                point=next(iter(cal))
                if case=='missing_point':cal.pop(point)
                elif case=='finite':cal[point]['max']=self.np.inf
                elif case=='scalar_count':cal[point]['count']=0
                elif case=='observations':source['count']=999
                elif case=='duplicate':source['indices'][1]=source['indices'][0]
                elif case=='label':source['entries'][0]['label']=10
                elif case=='shard':source['entries'][0]['row']=1
                elif case=='norm':source['normalization']['mean'][0]=0.
                elif case=='policy':info.threshold={'acc_drop_pp':1.}
                else:info.inputs[0]['kind']='image'
                if case in ('observations','duplicate','label','shard','norm'):
                    r14['worker']['calibration_source']=copy.deepcopy(source)
                self.assertFalse(self.worker.validate_source(r14,r13,info,cal,source)['passed'])

    def candidate(self,f):
        return self.parse(self.rule.plan(f.info.graph,f.calibration,deg=63,margin=1.5,slots=16384,
             placement='depth-dp',refresh='compiler',act_method='cheb'))

    def test_graph_names_map_by_activation_order_and_only_domain_changes(self):
        f=self.fixture();old=self.parse(f.original_text);new=self.candidate(f)
        checked=self.worker.check_candidate(f.info,old,new,f.calibration)
        self.assertTrue(checked['passed'],checked)
        graph_acts=[n for n in f.info.graph.nodes if n.op=='act']
        old_acts=[op for op in old.ops if op.kind=='act'];new_acts=[op for op in new.ops if op.kind=='act']
        self.assertEqual(len(new_acts),19)
        self.assertNotEqual(graph_acts[0].inputs[0],old_acts[0].inputs[0])
        for node,original,candidate in zip(graph_acts,old_acts,new_acts):
            cached=f.calibration[node.inputs[0]]
            self.assertEqual(original.attrs['domain'],(cached['min']*1.25,cached['max']*1.25))
            self.assertEqual(candidate.attrs['domain'],(cached['min']*1.5,cached['max']*1.5))
        self.assertEqual(new_acts[1].attrs['domain'],(-4.5,1.5))

    def test_candidate_rejects_extra_semantic_changes_but_ignores_source_positions(self):
        f=self.fixture();old=self.parse(f.original_text);new=self.candidate(f)
        moved=replace(new,header=replace(new.header,lineno=999),
                      params=tuple(replace(p,lineno=999) for p in new.params),
                      ops=tuple(replace(op,lineno=999,call_lines=(999,)) for op in new.ops))
        self.assertTrue(self.worker.check_candidate(f.info,old,moved,f.calibration)['passed'])
        for case in ('name','slots','output','kind','degree','domain','act_order','non_act','macro_stack'):
            with self.subTest(case=case):
                changed=copy.deepcopy(new);ops=list(changed.ops);indices=[i for i,o in enumerate(ops) if o.kind=='act']
                if case=='name':changed=replace(changed,name='different_function')
                elif case=='slots':changed=replace(changed,header=replace(changed.header,slots=8192))
                elif case=='output':changed=replace(changed,outputs=(changed.params[0].name,))
                elif case=='kind':ops[indices[0]]=replace(ops[indices[0]],kind='add')
                elif case=='degree':ops[indices[0]].attrs['deg']=31
                elif case=='domain':ops[indices[0]].attrs['domain']=(-3.1,3.)
                elif case=='act_order':ops[indices[0]],ops[indices[1]]=ops[indices[1]],ops[indices[0]]
                elif case=='non_act':next(op for op in ops if op.kind=='conv2d').attrs['algo']='other'
                else:ops[indices[0]]=replace(ops[indices[0]],macro_stack=('unexpected_macro',))
                if case not in ('name','slots','output'):changed=replace(changed,ops=tuple(ops))
                self.assertFalse(self.worker.check_candidate(f.info,old,changed,f.calibration)['passed'])

    def test_small_candidate_count_uses_one_backend_and_reports_new_arithmetic(self):
        from pt_agent import backends

        f=self.fixture(act_count=1);program=self.candidate(f)
        backend_type=backends.CountBackend
        with patch.object(backends,'CountBackend',wraps=backend_type) as backend:
            measured=self.worker._count_candidate(program,f.info.graph.weights)
        backend.assert_called_once_with(slots=16384)
        self.assertGreater(measured['stats']['ops_total'],0)
        self.assertEqual(measured['stats']['outputs'],1)
        self.assertIn('output_layouts',measured)

    def test_strict_math_boundary_original_dtype_shape_and_finite_are_unchanged(self):
        previous=self.worker.validation
        valid=self.np.zeros((1,10),dtype=self.np.float64);observed=valid.copy();observed[0,0]=1e-12
        self.assertTrue(previous.check_math(valid,observed)['passed'])
        observed[0,0]=self.np.nextafter(1e-12,self.np.inf)
        self.assertFalse(previous.check_math(valid,observed)['passed'])
        for wrong in (valid.ravel(),valid.astype(self.np.float32),self.np.full((1,10),self.np.nan),self.np.full((1,10),self.np.inf)):
            self.assertFalse(previous.check_math(valid,wrong)['passed'])

    def test_explicit_cached_paths_reject_cross_run_and_unrelated_model_info(self):
        f=self.fixture()
        source=self.worker._source_paths(f.args,f.r14,f.r13)
        self.assertEqual(source['original_program'],str(f.args.original_program.resolve()))
        self.assertFalse(source['included_in_current_turnaround'])
        for field in ('calibration','calibration_source','original_program','model_info'):
            with self.subTest(field=field):
                args=copy.copy(f.args);setattr(args,field,self.directory/('different-'+field))
                with self.assertRaises(self.worker.WorkerFailure):
                    self.worker._source_paths(args,f.r14,f.r13)
        wrong=copy.deepcopy(f.r14);wrong['worker']['r13_source_result']=str(self.directory/'other-result.json')
        with self.assertRaises(self.worker.WorkerFailure):
            self.worker._source_paths(f.args,wrong,f.r13)

    def fake_inputs(self,f):
        np=self.np
        arrays={'camera_features':np.broadcast_to(np.arange(1000,dtype=np.float64).reshape(1000,1,1,1),(1000,3,32,32))}
        labels=np.zeros(1000,dtype=np.int64)
        source=dict(records=str(f.args.records.resolve()),split='heldout',source_split='test',
                    indices=list(range(9000,10000)),sample_ids=['test:'+str(i) for i in range(9000,10000)],
                    labels=labels.tolist(),count=1000,batch=1,normalization=f.info.source['normalization'],
                    shard='test-09000.records',confirm_samples=0,report_samples=0)
        return arrays,labels,source

    def invoke(self,f,*,domain_value=0.,wrong_predictions=0,error_engine=None,error_index=0,
               bad_dtype=False,math_pair=None,count_error=False):
        """Run new orchestration on small fixtures and fake outputs only."""
        from pt_agent import analyzer,backends,config,cost,tensor_sim,toolchain
        from pt_agent.intake import probe

        np=self.np;order=[];mocks={}
        arrays,labels,sample_source=self.fake_inputs(f)
        original_write=self.worker.write_json

        def checkpoint(path,value):
            # Keep actual 1000 loops/events and full final output; avoid repeatedly
            # serializing the growing fake observation list between every engine.
            if Path(path).name!='worker_result.json' or (
                    value.get('case_budget_observations') and
                    value['case_budget_observations'][-1].get('boundary')=='terminal'):
                original_write(path,value)

        def index_of(image):return int(image[0,0,0,0])

        def logits(index,*,program=False):
            out=np.zeros((1,10),dtype=np.float64);out[0,0]=2.25 if program else 2.
            if program and index<wrong_predictions:out[0,1]=3.
            return out

        def fail(engine,index):
            if error_engine==engine and index==error_index:
                raise RuntimeError('synthetic engine failure')

        def reference(graph,inputs,*,keep):
            self.assertFalse(keep);index=index_of(inputs['camera_features'])
            order.append((index,'reference_forward'));fail('reference_forward',index)
            return {'declared_logits':logits(index)}

        def tensor(program,weights,image,*,cache,act_mode,observer=None):
            self.assertFalse(cache);index=index_of(image);engine='tensor_'+act_mode
            order.append((index,engine));fail(engine,index)
            out=logits(index,program=act_mode=='program')
            if act_mode=='program':
                self.assertIsNotNone(observer)
                for op in program.ops:
                    if op.kind=='act':
                        observer(op.inputs[0],np.array([domain_value],dtype=np.float64),
                                 ('cheb',63,*op.attrs['domain']))
                if bad_dtype and index==error_index:return out.astype(np.float32)
                if error_engine=='nonfinite' and index==error_index:out[0,0]=np.nan
            if math_pair=='tensor_exact' and index==0 and act_mode=='exact':out[0,0]+=1e-8
            return out

        def packed(program,weights,image,*,act_mode):
            index=index_of(image);engine='mock_'+act_mode
            order.append((index,engine));fail(engine,index)
            out=logits(index,program=act_mode=='program')
            if math_pair=='mock_program' and index==0 and act_mode=='program':out[0,0]+=1e-8
            return out

        counted=dict(stats=dict(f.saved_stats,distinct_consts=7),output_layouts=[dict(shape=[1,10])])
        forbidden=ModuleType('pt_agent.llm')
        forbidden.LLMClient=Mock(side_effect=AssertionError('LLM is forbidden'))
        forbidden.Provider=Mock(side_effect=AssertionError('Provider is forbidden'))
        with ExitStack() as stack:
            stack.enter_context(patch.dict('sys.modules',{'pt_agent.llm':forbidden}))
            stack.enter_context(patch.object(self.worker,'write_json',side_effect=checkpoint))
            stack.enter_context(patch.object(self.worker.validation,'write_json',side_effect=checkpoint))
            mocks['heldout']=stack.enter_context(patch.object(self.worker,'_read_heldout',return_value=(arrays,labels,sample_source)))
            mocks['plan']=stack.enter_context(patch.object(self.worker,'_plan_candidate',wraps=self.worker._plan_candidate))
            mocks['count']=stack.enter_context(patch.object(self.worker,'_count_candidate',
                side_effect=RuntimeError('synthetic Count failure') if count_error else None,return_value=counted))
            mocks['reference']=stack.enter_context(patch.object(analyzer,'reference_forward',side_effect=reference))
            mocks['tensor']=stack.enter_context(patch.object(tensor_sim,'run_tensor',side_effect=tensor))
            mocks['packed']=stack.enter_context(patch.object(self.worker,'_packed_logits',side_effect=packed))
            for module,name in ((analyzer,'calibrate'),(probe,'import_torch'),(backends,'EmitBackend'),
                                (cost,'estimate'),(config,'load'),(toolchain,'trace'),
                                (toolchain,'compile'),(toolchain,'execute_cpu')):
                mocks['forbidden:'+name]=stack.enter_context(patch.object(module,name,
                    side_effect=AssertionError(name+' is forbidden')))
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                exitcode=self.worker.main(f.cli)
        for name,mock in mocks.items():
            if name.startswith('forbidden:'):mock.assert_not_called()
        forbidden.LLMClient.assert_not_called();forbidden.Provider.assert_not_called()
        result=json.loads((f.args.work_dir/'worker_result.json').read_text())
        return SimpleNamespace(result=result,exitcode=exitcode,mocks=mocks,order=order)

    def assert_success_counts(self,observed):
        result=observed.result
        self.assertEqual(observed.exitcode,0,result.get('error'))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['completed_samples'],1000)
        self.assertEqual(result['heldout_samples'],1000)
        for key in self.worker.STAGE_FIELDS:self.assertIs(result[key],True,key)
        for key in self.worker.COUNT_FIELDS:self.assertEqual(result[key],1,key)
        for key in self.worker.ZERO_FIELDS:self.assertEqual(result[key],0,key)
        for key in self.worker.UNMEASURED_FIELDS:self.assertIsNone(result[key],key)
        for key in self.worker.CACHE_NAMES:
            self.assertEqual(result['cached_source_reads'][key],dict(attempted=1,completed=1))
        for key in self.worker.ENGINE_NAMES:
            count=1000 if key in ('reference_forward','tensor_program') else 3
            self.assertEqual(result['engine_counts'][key],dict(attempted=count,completed=count))
        self.assertFalse(result['candidate_frozen'])
        self.assertTrue(result['evaluation_completed']);self.assertFalse(result['outputs_partial'])
        observed.mocks['plan'].assert_called_once();observed.mocks['count'].assert_called_once()
        observed.mocks['heldout'].assert_called_once()

    def test_complete_new_candidate_uses_fixed_plan_one_count_and_nine_math_pairs(self):
        f=self.fixture();observed=self.invoke(f);self.assert_success_counts(observed);result=observed.result
        self.assertTrue(result['task_passed']);self.assertTrue(result['domain_passed']);self.assertTrue(result['heldout_acceptance'])
        self.assertEqual(result['rule_options'],self.worker.RULE_OPTIONS)
        self.assertEqual(result['source_count_stats'],f.saved_stats)
        self.assertEqual(result['count_stats']['distinct_consts'],7)
        self.assertEqual(result['max_logit_absolute_difference'],.25)
        self.assertEqual(len(result['math_probes']),9)
        self.assertTrue(all(row['passed'] for row in result['math_probes']))
        for index in range(3):
            self.assertEqual([name for sample,name in observed.order if sample==index],
                ['reference_forward','tensor_exact','mock_exact','tensor_program','mock_program'])
        self.assertEqual([name for sample,name in observed.order if sample==3],['reference_forward','tensor_program'])
        self.assertTrue(all(row['count']==1000 and row['deg']==63 for row in result['domain_rows']))
        self.assertEqual(f.args.original_program.read_text(),f.original_text)
        self.assertNotEqual((f.args.work_dir/'program.ptdsl').read_text(),f.original_text)
        differences=json.loads(Path(result['artifacts']['count_differences']).read_text())
        self.assertTrue(differences['diagnostic_only'])
        self.assertEqual(differences['differences']['distinct_consts'],dict(source=2,candidate=7))
        self.assertEqual(result['sample_source']['indices'],list(range(9000,10000)))

    def test_complete_domain_rejection_is_normal_completed_quality_result(self):
        f=self.fixture('vgg16_silu_avgpool');observed=self.invoke(f,domain_value=3.1)
        self.assert_success_counts(observed);result=observed.result
        self.assertTrue(result['task_passed']);self.assertFalse(result['domain_passed'])
        self.assertFalse(result['heldout_acceptance']);self.assertGreater(result['max_u'],.90)
        self.assertEqual(result['source_status'],'artifact_budget_exceeded')

    def test_complete_task_rejection_preserves_point_five_percentage_point_policy(self):
        f=self.fixture('alexnet_silu_avgpool');observed=self.invoke(f,wrong_predictions=6)
        self.assert_success_counts(observed);result=observed.result
        self.assertFalse(result['task_passed']);self.assertTrue(result['domain_passed'])
        self.assertFalse(result['heldout_acceptance'])
        self.assertEqual(result['threshold'],dict(acc_drop_pp=.5))
        self.assertAlmostEqual(result['validation_metrics']['acc_drop_pp'],.6)
        self.assertEqual(result['acc_drop_pp_unit'],'percentage points')

    def test_original_task_policy_accepts_exact_point_five_pp_without_rounding(self):
        from pt_agent.metrics import compare,passes

        exact=self.np.zeros((1000,10),dtype=self.np.float64);exact[:,0]=2.
        candidate=exact.copy();candidate[:5,1]=3.
        measured=compare(dict(values=exact,task='classification'),candidate,self.np.zeros(1000,dtype=self.np.int64))
        self.assertEqual(measured['acc_drop_pp'],.5);self.assertTrue(passes(measured,dict(acc_drop_pp=.5)))
        candidate[5,1]=3.;measured=compare(dict(values=exact,task='classification'),candidate,self.np.zeros(1000,dtype=self.np.int64))
        self.assertFalse(passes(measured,dict(acc_drop_pp=.5)))

    def test_first_exact_probe_error_stops_before_program_and_leaves_quality_unknown(self):
        f=self.fixture();observed=self.invoke(f,math_pair='tensor_exact');result=observed.result
        self.assertEqual(observed.exitcode,1);self.assertEqual(result['status'],'math_failed')
        self.assertEqual(result['completed_samples'],0);self.assertIs(result['math_probes_passed'],False)
        self.assertEqual(result['engine_counts']['tensor_exact'],dict(attempted=1,completed=1))
        self.assertEqual(result['engine_counts']['tensor_program'],dict(attempted=0,completed=0))
        self.assertIsNone(result['heldout_acceptance']);self.assertFalse(result['evaluation_completed'])

    def test_program_mock_probe_error_retains_one_complete_sample_and_partial_outputs(self):
        f=self.fixture();observed=self.invoke(f,math_pair='mock_program');result=observed.result
        self.assertEqual(result['status'],'math_failed');self.assertEqual(result['completed_samples'],1)
        self.assertEqual(result['heldout_samples'],1);self.assertEqual(len(result['math_probes']),3)
        self.assertTrue(result['outputs_partial']);self.assertIsNone(result['validation_metrics'])
        self.assertEqual(result['engine_counts']['mock_program'],dict(attempted=1,completed=1))

    def test_runtime_failure_preserves_attempted_completed_counts_and_previous_three(self):
        f=self.fixture();observed=self.invoke(f,error_engine='tensor_program',error_index=3);result=observed.result
        self.assertEqual(result['status'],'error');self.assertEqual(result['completed_samples'],3)
        self.assertEqual(result['engine_counts']['reference_forward'],dict(attempted=4,completed=4))
        self.assertEqual(result['engine_counts']['tensor_program'],dict(attempted=4,completed=3))
        self.assertTrue(result['math_probes_passed']);self.assertIsNone(result['max_u'])
        self.assertIsNone(result['task_passed']);self.assertIsNone(result['domain_passed'])

    def test_nonfinite_and_original_dtype_fail_before_accepting_fourth_sample(self):
        for bad_dtype in (False,True):
            with self.subTest(bad_dtype=bad_dtype):
                f=self.fixture('alexnet_silu_avgpool' if bad_dtype else 'vgg16_silu_avgpool')
                observed=self.invoke(f,error_engine=None if bad_dtype else 'nonfinite',error_index=3,bad_dtype=bad_dtype)
                result=observed.result
                self.assertEqual(result['status'],'shape_failed' if bad_dtype else 'nonfinite')
                self.assertEqual(result['completed_samples'],3)
                self.assertEqual(result['engine_counts']['tensor_program'],dict(attempted=4,completed=4))
                self.assertIsNone(result['heldout_acceptance'])
                if not bad_dtype:self.assertIs(result['all_finite'],False)

    def test_source_and_count_failures_preserve_unreached_null_stages_and_no_forwards(self):
        for broken_source in (False,True):
            with self.subTest(broken_source=broken_source):
                f=self.fixture('vgg16_silu_avgpool' if broken_source else 'resnet20_silu')
                if broken_source:
                    f.r14['worker']['reference_passed']=False;self.write_json(f.args.r14_result,f.r14)
                observed=self.invoke(f,count_error=not broken_source);result=observed.result
                self.assertEqual(observed.exitcode,1);self.assertFalse(result['evaluation_completed'])
                self.assertEqual(result['status'],'source_failed' if broken_source else 'error')
                self.assertIsNone(result['samples_checked']);self.assertEqual(result['completed_samples'],0)
                self.assertIsNone(result['math_probes_passed']);observed.mocks['heldout'].assert_not_called()
                if broken_source:
                    observed.mocks['plan'].assert_not_called();observed.mocks['count'].assert_not_called()
                else:
                    self.assertEqual(result['count_attempts'],1);self.assertEqual(result['count_completions'],0)
                self.assertTrue(all(value==dict(attempted=0,completed=0) for value in result['engine_counts'].values()))

    def test_real_heldout_reader_requests_only_fixed_heldout_and_preserves_features(self):
        from pt_agent.data import samples

        f=self.fixture();arrays,labels,source=self.fake_inputs(f)
        self.write_json(f.args.records/'dataset.json',f.calibration_source['dataset_metadata'])
        with patch.object(self.cifar,'RECORDS',f.args.records),patch.object(samples,'LegacyCifar') as legacy:
            legacy.return_value.get.return_value=arrays,labels
            actual_arrays,actual_labels,actual_source=self.worker._read_heldout(f.args,f.info)
        legacy.return_value.get.assert_called_once_with('heldout',1000)
        legacy.assert_called_once_with(f.info.graph,normalization=(list(self.cifar.MEAN),list(self.cifar.STD)))
        self.assertIs(actual_arrays,arrays);self.assertIs(actual_labels,labels)
        self.assertEqual(actual_source,source);self.assertEqual(f.info.inputs[0]['kind'],'features')

    def test_budget_phase_preserves_real_partial_record_when_limit_is_exceeded(self):
        f=self.fixture();f.args.work_dir.mkdir(parents=True)
        recorder=self.worker.StageRecorder(f.args.work_dir/'events.jsonl');result=self.worker._initial_result()
        with patch.object(self.worker.original,'_case_size',return_value=self.worker.CASE_BYTES+1):
            with self.assertRaises(self.worker.WorkerFailure) as caught:
                with self.worker._phase(f.args,recorder,result,'tiny-check'):
                    self.fail('over-budget phase body must stay uncalled')
        self.assertEqual(caught.exception.status,'artifact_budget_exceeded')
        self.assertEqual(result['case_budget_observations'][-1]['boundary'],'before')
        self.assertFalse(result['case_budget_observations'][-1]['within_budget'])
