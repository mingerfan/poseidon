"""U3b profile dispatch, frozen v4 prompts, and a real offline CNN loop."""

from contextlib import ExitStack
from dataclasses import asdict
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from pt_agent.analyzer import calibrate
from pt_agent.dsl.parser import parse
from pt_agent.experiments import u3b_agent_cnn as experiment
from pt_agent.intake import import_module
from pt_agent.models.zoo import get
from pt_agent.paths import RESULTS_DIR
from pt_agent.planners import agent, agent_tools, confirm, prompts, tuner
from pt_agent.planners.rule import plan
from pt_agent.tensor_sim import clear_caches
from pt_agent.tests._artifacts import DATA, require
from pt_agent.tests.test_r8h_tune_all import FakeClient, feedback, source, tune_result


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(1,2,1,bias=False)
        self.pool=nn.MaxPool2d(2)
        self.fc=nn.Linear(16,2)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([.5,1.]).reshape(2,1,1,1))
            self.fc.weight.fill_(.001)
            self.fc.bias.copy_(torch.tensor([10.,0.]))

    def forward(self,x):
        y=self.conv(x)
        y=torch.cat((y,y),dim=1)
        return self.fc(self.pool(y).flatten(1))


def fixture():
    images=np.sin(np.arange(6*16,dtype=np.float64)*.19).reshape(6,1,4,4)
    graph=import_module(SmallCNN().double().eval(),(torch.from_numpy(images[:1]),),
                        'small_pool_concat',frontend='aten')
    calibration=calibrate(graph,{graph.inputs[0]:images})
    with patch.object(agent_tools,'heldout_set',return_value=(images,np.zeros(6,dtype=np.int64))):
        ctx=agent_tools.build_context(graph,calibration,n_val=6)
    ctx.prompt_version='r8-v5'
    ctx.val_split='confirm'
    return ctx,plan(graph,calibration,deg=127,margin=1.5)


class EngineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        clear_caches()

    def test_default_root_dispatch_is_unchanged(self):
        program=source(1)
        with TemporaryDirectory() as directory, \
                patch.object(agent,'build_system_prompt',return_value='offline'), \
                patch.object(agent,'initial_message',return_value='offline'), \
                patch.object(agent,'evaluate',return_value=feedback(100)) as evaluation, \
                patch.object(agent,'tune',return_value=tune_result(program,100,90)) as tuning, \
                patch.object(agent,'confirm_select',return_value=dict(selected=None,evals=[])) as confirmation:
            agent.run_agent(object(),FakeClient([program]),max_iters=1,trace_path=Path(directory)/'trace.jsonl')
        self.assertEqual(evaluation.call_args.kwargs,{})
        self.assertNotIn('engine_profile',tuning.call_args.kwargs)
        self.assertEqual(confirmation.call_args.kwargs,{})

    def test_tensor_profile_is_restricted_to_v5(self):
        with patch.object(agent,'build_system_prompt') as build:
            with self.assertRaisesRegex(ValueError,'r8-v5'):
                agent.run_agent(SimpleNamespace(prompt_version='r8-v4'),object(),
                    trace_path=Path('unused'),engine_profile='tensor')
        build.assert_not_called()

    def test_cnn_full_loop_all_approximate_evaluations_tensor(self):
        ctx,text=fixture()
        self.assertIn('maxpool(',text)
        self.assertIn('concat(',text)
        observed={}
        with TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(agent_tools,'MockBackend',side_effect=AssertionError('tensor profile used Mock')))
            for label,module in (('evaluate',agent_tools),('tune',tuner),('confirm',confirm)):
                observed[label]=stack.enter_context(patch.object(module,'validate',wraps=module.validate))
            result=agent.run_agent(ctx,FakeClient([text]),max_iters=1,
                trace_path=Path(directory)/'trace.jsonl',engine_profile='tensor')
            self.assertEqual(result['status'],'feasible')
            self.assertEqual(result['tune_policy'],'new_best')
            checked=experiment.report_evaluation(parse(result['best_program']),ctx,engine_profile='tensor')
            self.assertIsNone(checked['boot_input_max_abs'])
            rows=[json.loads(line) for line in (Path(directory)/'trace.jsonl').read_text().splitlines()]
        for name,spy in observed.items():
            self.assertGreater(spy.call_count,0,name)
            self.assertTrue(all(c.kwargs.get('engine','mock')=='tensor' for c in spy.call_args_list),name)
        self.assertIsNone(rows[0]['fb']['validation']['boot_input_max_abs'])
        self.assertIsNone(rows[0]['tuned']['history'][-1]['boot_input_max_abs'])

    def test_feedback_field_names_match_and_boot_value_is_null(self):
        ctx,text=fixture()
        mock=agent_tools.evaluate(text,ctx)
        tensor=agent_tools.evaluate(text,ctx,engine='tensor')
        self.assertEqual(set(mock),set(tensor))
        self.assertEqual(set(mock['validation']),set(tensor['validation']))
        self.assertIsNone(tensor['validation']['boot_input_max_abs'])
        self.assertIsNotNone(mock['validation']['boot_input_max_abs'])

    def test_report_default_uses_mock(self):
        with patch.object(agent_tools,'validate',return_value={}) as validation:
            experiment.report_evaluation(object(),object())
        self.assertEqual(validation.call_args.kwargs,dict(stop_on_divergence=False))

    def test_v5_capabilities_and_approximation_points(self):
        ctx,_=fixture()
        text=prompts.build_system_prompt(ctx)
        for token in ('maxpool','avgpool','concat','batchnorm','groups','degrees','domain','linear'):
            self.assertIn(token,text)
        nodes=json.loads(prompts.initial_message(ctx))['nodes']
        pool=next(row for row in nodes if row['op']=='maxpool')
        self.assertIn('calib',pool)
        self.assertIn('needed_margin',pool)
        self.assertEqual(prompts.PROMPT_VERSION,'r8-v4')

    def test_mock_spot_captures_logits_once_and_matches(self):
        ctx,text=fixture()
        with TemporaryDirectory() as directory:
            report=experiment.mock_spot(parse(text),ctx,Path(directory),'agent','heldout',6)
            self.assertTrue(report['complete'])
            self.assertTrue(report['classes_equal'] and report['divergence_equal'])
            self.assertLessEqual(report['max_normalized_diff'],1e-9)
            with patch.object(agent_tools,'validate',side_effect=AssertionError('do not repeat audit')):
                self.assertEqual(experiment.mock_spot(parse(text),ctx,Path(directory),'agent','heldout',6),report)

    def test_mock_spot_mismatch_stops_and_is_retained(self):
        ctx,text=fixture()
        def changed(*args,engine,output_callback,**kwargs):
            output_callback(np.full((6,2),1. if engine=='mock' else 0.))
            return dict(diverged_indices=[])
        with TemporaryDirectory() as directory,patch.object(agent_tools,'validate',side_effect=changed):
            path=Path(directory)
            with self.assertRaisesRegex(RuntimeError,'mismatch'):
                experiment.mock_spot(parse(text),ctx,path,'rule','report',6)
            self.assertFalse(json.loads((path/'rule-report-spot.json').read_text())['complete'])

    def test_order_and_finite_budget(self):
        self.assertEqual(experiment.ORDER,('resnet20_relu','squeezenet_silu_avgpool','squeezenet_relu_maxpool',
            'mobilenet_silu','mobilenet_relu','vgg16_silu_avgpool','vgg16_relu_maxpool',
            'alexnet_silu_avgpool','alexnet_relu_maxpool'))
        self.assertEqual((experiment.API_BUDGET,experiment.GLOBAL_LIMIT),(50,113))

    def test_divergence_mismatch_is_recorded_before_stopping(self):
        ctx,text=fixture()
        def changed(*args,engine,output_callback,**kwargs):
            outputs=np.zeros((6,2))
            if engine=='mock':outputs[0]=np.nan
            output_callback(outputs)
            return dict(diverged_indices=[0] if engine=='mock' else [])
        with TemporaryDirectory() as directory,patch.object(agent_tools,'validate',side_effect=changed):
            path=Path(directory)
            with self.assertRaisesRegex(RuntimeError,'mismatch'):
                experiment.mock_spot(parse(text),ctx,path,'agent','report',6)
            retained=json.loads((path/'agent-report-spot.json').read_text())
            self.assertFalse(retained['divergence_equal'])
            self.assertEqual(retained['diverged_indices'],dict(tensor=[],mock=[0]))

    def test_completed_trace_recovers_confirmation_without_generation_or_tuning(self):
        ctx,text=fixture()
        fb=agent_tools.evaluate(text,ctx,engine_profile='tensor')
        result=tuner.tune(parse(text),ctx,budget_s=2,max_sweeps=1,engine_profile='tensor')
        tuned={k:v for k,v in asdict(result).items() if k!='program'}
        with TemporaryDirectory() as directory:
            case=Path(directory)
            rows=[dict(iteration=i,messages=[dict(role='assistant',content=json.dumps(dict(program=text)))],
                fb=fb,tuned=tuned if i==1 else None,usage=dict(prompt_tokens=1,completion_tokens=2),seconds=.1)
                for i in range(1,7)]
            (case/'agent_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with patch.object(agent,'run_agent',side_effect=AssertionError('no generation replay')), \
                    patch.object(tuner,'tune',side_effect=AssertionError('no retuning')), \
                    patch('pt_agent.llm.LLMClient.chat',side_effect=AssertionError('no LLM call')):
                restored=experiment.complete_retained_confirmation(ctx,case,prior_seconds=10.)
            self.assertEqual(restored['status'],'feasible')
            self.assertEqual(restored['llm_calls'],6)
            self.assertEqual(restored['pool_size'],7 if result.feasible else 6)
            self.assertEqual(restored['prompt_tokens'],6)
            self.assertEqual(restored['completion_tokens'],12)
            self.assertFalse(restored['regenerated'] or restored['retuned'])
            self.assertEqual(restored['best_program'],restored['confirm']['selected'].source)
            with self.assertRaisesRegex(RuntimeError,'no repeat'):
                experiment.complete_retained_confirmation(ctx,case,prior_seconds=10.)

    def test_context_retains_heldout_and_binds_lazy_cifar_for_later_splits(self):
        from pt_agent.data.samples import LegacyCifar
        info=SimpleNamespace(graph=object(),input_handling=dict(internal_regression_source='legacy_cifar'))
        retained=SimpleNamespace(val_images=object())
        with TemporaryDirectory() as directory, \
                patch('pt_agent.models.info.ModelInfo.load',return_value=info), \
                patch('pt_agent.data.samples.StoredSplits') as stored, \
                patch('pt_agent.data.samples.LegacyCifar.get',side_effect=AssertionError('later split loaded early')), \
                patch.object(agent_tools,'build_context',return_value=retained) as build:
            case=Path(directory)
            (case/'ACCEPTANCE.json').write_text('{}')
            (case/'calibration.json').write_text('{}')
            ctx=experiment.context(case)
            self.assertIs(ctx,retained)
            self.assertIs(build.call_args.kwargs['samples'],stored.return_value)
            self.assertIsInstance(ctx.samples,LegacyCifar)
            self.assertIs(ctx.samples.graph,info.graph)
            self.assertEqual(ctx.prompt_version,'r8-v5')

    def test_saved_agent_releases_caches_before_external_work_and_clears_prior_error(self):
        ctx,text=fixture()
        with TemporaryDirectory() as directory:
            case=Path(directory)/'case';case.mkdir()
            prior=Path(directory)/'prior';prior.mkdir()
            (prior/'program.ptdsl').write_text(text)
            (case/'AGENT.json').write_text(json.dumps(dict(status='confirm_failed',confirm=dict(selected=None),best_program=None)))
            (case/'REPORT.json').write_text(json.dumps(dict(complete=False,error='MemoryError: prior',spots={},report_metrics={})))
            with patch.object(experiment,'context',return_value=ctx), \
                    patch.object(experiment,'split_context',return_value=ctx), \
                    patch.object(experiment,'mock_spot',return_value=dict(complete=True)), \
                    patch.object(confirm,'build_confirm_context',return_value=ctx), \
                    patch('pt_agent.tensor_sim.clear_caches',wraps=clear_caches) as clearing, \
                    patch('pt_agent.llm.LLMClient.chat',side_effect=AssertionError('saved Agent is retained')), \
                    patch.object(experiment,'report_evaluation') as reporting:
                previous=[]
                def checked(*args,**kwargs):
                    self.assertGreater(clearing.call_count,previous[-1] if previous else 0)
                    previous.append(clearing.call_count)
                    return dict(top1=100.)
                reporting.side_effect=checked
                result=experiment.model_inside(SimpleNamespace(out=case,source_case=prior,model='offline'))
                self.assertTrue(result['complete'])
                self.assertNotIn('error',result)
                self.assertGreaterEqual(clearing.call_count,2)


FROZEN=RESULTS_DIR/'r8e7-reconfirm-20261022T122558/models'
PROMPTS=RESULTS_DIR/'u3a-compat-20261024T152017685800/prompts'


@require(DATA,RESULTS_DIR/'models-v2',FROZEN,PROMPTS)
class FrozenV4Tests(unittest.TestCase):
    def test_four_frozen_prompt_texts_unchanged(self):
        for name in ('resnet20','resnet20_silu','cnn_plain','mlp'):
            with self.subTest(model=name):
                model=get(name,RESULTS_DIR/'models-v2')
                ctx=agent_tools.build_context(model.import_graph(),
                    json.loads((FROZEN/name/'calibration.json').read_text()),n_val=20,normalization=model.normalization)
                self.assertEqual(prompts.build_system_prompt(ctx),(PROMPTS/name/'r8-v4.system.txt').read_text())
                self.assertEqual(prompts.initial_message(ctx),(PROMPTS/name/'r8-v4.initial.json').read_text())
                del ctx,model
                gc.collect()


if __name__=='__main__':
    unittest.main()
