"""A1 §6: public interfaces, kernel capabilities and domain provenance."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent import domain_criteria as criteria
from pt_agent.dsl.interface import check_interface, require_interface, inputs, SIGNATURE_RULE
from pt_agent.dsl.parser import parse
from pt_agent.dsl.typecheck import check
from pt_agent.dsl.constraints import CONSTRAINTS, SHAPE_RULES, for_op
from pt_agent.dsl.methods import supported_methods, method_table
from pt_agent.dsl.ops_spec import OPS
from pt_agent.kernels import approx, poly
from pt_agent.planners import agent, agent_tools, confirm, prompts, transformer_tuner
from pt_agent.planners.transformer_session import ModelBudgetClient, run_session, OUTPUT_TOKENS
from pt_agent.tests.test_a1_agent import fixture, SOURCE, RecordedClient
from pt_agent.tests._artifacts import COMPILER, RESULTS_DIR, require


def context():
    info, calibration, samples = fixture()
    ctx = agent_tools.build_context(info, calibration, samples=samples)
    ctx.prompt_version = 'r8-v8'
    return ctx


class InterfaceTests(unittest.TestCase):
    def test_signature_order_name_kind_shape_and_missing_extra(self):
        info, _, _ = fixture()
        original = parse(SOURCE)
        self.assertTrue(check_interface(original, info)['ok'])
        variants = [SOURCE.replace('x:', 'v:').replace('pack(x,', 'pack(v,'),
                    SOURCE.replace('x: Cipher', 'x: Mask'),
                    SOURCE.replace('Cipher[1,1,2]', 'Cipher[1,2,2]'),
                    SOURCE.replace('x: Cipher[1,1,2]', 'x: Cipher[1,1,2], m: Mask[1,1]')]
        for source in variants:
            result = check_interface(parse(source), info)
            self.assertFalse(result['ok'])
            self.assertIn('expected', result['errors'][0])
            self.assertIn('actual', result['errors'][0])

    def test_mask_expected_type_and_forward_order(self):
        ctx = context()
        ctx.model_info.inputs += [dict(name='m', kind='mask', shape=[1,1])]
        source = SOURCE.replace('x: Cipher[1,1,2]', 'x: Cipher[1,1,2], m: Cipher[1,1]')
        fb = agent_tools.evaluate(source, ctx, engine_profile='tensor')
        self.assertEqual(fb['stage'], 'interface')
        self.assertEqual(fb['errors'][0]['expected']['dsl_type'], 'Mask')
        reversed_source = SOURCE.replace('x: Cipher[1,1,2]', 'm: Mask[1,1], x: Cipher[1,1,2]')
        self.assertFalse(check_interface(parse(reversed_source), ctx.model_info)['ok'])

    def test_interface_precedes_every_execution_and_typecheck(self):
        ctx = context()
        renamed = SOURCE.replace('x:', 'v:').replace('pack(x,', 'pack(v,')
        with patch.object(agent_tools, 'check', side_effect=AssertionError('typecheck reached')):
            result = agent_tools.evaluate(renamed, ctx, engine_profile='tensor')
        self.assertEqual(result['stage'], 'interface')
        self.assertFalse(result['feasible'])
        self.assertIsNone(result['exact'])
        with self.assertRaisesRegex(ValueError, 'Input interface'):
            transformer_tuner.tune(parse(renamed), ctx)
        with self.assertRaisesRegex(ValueError, 'Input interface'):
            confirm.confirm_select([confirm.Candidate(renamed,'llm',1,1,0)], ctx)

    def test_report_export_guards_before_execution_or_sample_read(self):
        from pt_agent.experiments.a1_transformer_agent import evaluate_frozen
        from pt_agent.export import export_package
        ctx = context()
        renamed = SOURCE.replace('x:', 'v:').replace('pack(x,', 'pack(v,')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); program = root/'program.ptdsl'; program.write_text(renamed)
            marker = root/'FROZEN.json'; marker.write_text(json.dumps(dict(program=str(program))))
            with self.assertRaisesRegex(ValueError, 'Input interface'):
                evaluate_frozen(ctx.model_info, ctx.calibration, ctx.samples, program, root/'report',
                                frozen=marker, prompt_version='r8-v8')
            with self.assertRaisesRegex(ValueError, 'Input interface'):
                export_package(renamed,parse(renamed),ctx.weights,root/'export',model_info=ctx.model_info,
                               prompt_version='r8-v8')
            with self.assertRaisesRegex(ValueError, 'original ModelInfo'):
                export_package(SOURCE,parse(SOURCE),ctx.weights,root/'export',prompt_version='r8-v8')
            self.assertFalse((root/'export').exists())
        self.assertNotIn('report', ctx.samples.reads)

    def test_summary_generated_from_model_info(self):
        ctx = context()
        ctx.model_info.inputs += [dict(name='m', kind='mask', shape=[1,1])]
        message = json.loads(prompts.initial_message(ctx))
        self.assertEqual(message['inputs'], inputs(ctx.model_info))
        self.assertEqual(message['inputs'][1],dict(name='m',kind='mask',shape=[1,1],
                         dsl_type='Mask',pack_layouts=['flat']))
        self.assertIn(SIGNATURE_RULE, prompts.build_system_prompt(ctx))


class MethodAndConstraintTests(unittest.TestCase):
    def test_kernel_generated_support(self):
        for fn in approx.REMEZ_FUNCTIONS:
            self.assertIn('remez', supported_methods(fn))
        for fn in OPS['act'].attrs['fn'].choices:
            self.assertEqual('cheb' in supported_methods(fn), fn in poly.EXACT)
            self.assertEqual('newton' in supported_methods(fn), fn in approx.NEWTON_FUNCTIONS)
        self.assertNotIn('inv', poly.EXACT); self.assertNotIn('rsqrt', poly.EXACT)

    def test_unsupported_method_is_typecheck_not_degenerate(self):
        ctx = context()
        source = SOURCE.replace('fn="tanh"', 'fn="inv"').replace('domain=(-1.,1.)','domain=(.1,1.)')
        with patch('pt_agent.planners.approximation_failures.program_failures',
                   side_effect=AssertionError('fitting reached')):
            fb = agent_tools.evaluate(source, ctx, engine_profile='tensor')
        self.assertEqual(fb['stage'], 'typecheck')
        error = next(item for item in fb['errors'] if item['code']=='E_METHOD_UNSUPPORTED')
        self.assertIn('newton',error['message']); self.assertIn('remez',error['message'])
        self.assertNotIn('degenerate',error['message'])
        from pt_agent.export import export_package
        with tempfile.TemporaryDirectory() as tmp:
            with patch('pt_agent.export.package.lower',side_effect=AssertionError('numerics reached')):
                with self.assertRaisesRegex(ValueError,'E_METHOD_UNSUPPORTED'):
                    export_package(source,parse(source),ctx.weights,Path(tmp)/'export',
                                   model_info=ctx.model_info,prompt_version='r8-v8')

    def test_norm_and_all_inverse_roles_reject_cheb(self):
        norm='''@fhe.program(slots=1024)
def f(x: Cipher[1,1,2]):
 a=pack(x,layout="token")
 y=layernorm(a,eps=1e-5,rsqrt_method="cheb",rsqrt_domain=(.1,2.),rsqrt_deg=7)
 return y
'''
        self.assertIn('E_METHOD_UNSUPPORTED',[d.code for d in check(parse(norm),{},prompt_version='r8-v8').diags])
        soft='''@fhe.program(slots=1024)
def f(x: Cipher[1,1,2]):
 a=pack(x,layout="token")
 y=softmax(a,steps=1,exp_method="cheb",exp_deg=7,exp_domain=(-1.,1.),inv_method="cheb",inv_deg=7,inv_domain=(.1,8.),step_inv_method="cheb",step_inv_deg=7,step_inv_domain=(.1,2.),final_inv_method="cheb",final_inv_deg=7,final_inv_domain=(.1,2.),inv_budget=.25)
 return y
'''
        errors=[d for d in check(parse(soft),{},prompt_version='r8-v8').diags if d.code=='E_METHOD_UNSUPPORTED']
        self.assertEqual(len(errors),3)

    def test_all_constraint_explanations_appear_and_no_target_weights(self):
        ctx=context(); text=prompts.build_system_prompt(ctx)
        for c in CONSTRAINTS:
            self.assertIn(c.explanation,text)
        for text_rule in SHAPE_RULES.values():
            self.assertIn(text_rule,text)
        for name in OPS:
            for explanation in for_op(name): self.assertIn(explanation,text)
        self.assertIn(method_table(),text)
        for name in ctx.weights: self.assertNotIn('W["'+name+'"]',text)

    def test_same_shape_public_add_and_mask_flat(self):
        source='''@fhe.program(slots=1024)
def f(x: Cipher[1,1,2], m: Mask[1,1]):
 a=pack(x,layout="token")
 am=pack(m,layout="flat")
 y=add(a,W["public"])
 return y
'''
        self.assertTrue(check(parse(source),{'public':(1,1,2)},prompt_version='r8-v8').ok)
        bad=source.replace('pack(m,layout="flat")','pack(m,layout="token")')
        self.assertIn('E_KIND',[d.code for d in check(parse(bad),{'public':(1,1,2)},prompt_version='r8-v8').diags])

    def test_legacy_typecheck_path_unchanged(self):
        source=SOURCE.replace('fn="tanh"','fn="inv"').replace('domain=(-1.,1.)','domain=(.1,1.)')
        self.assertTrue(check(parse(source),{'matrix':(2,2)}).ok)

    def test_v8_last_dimension_vector_bias_uses_explicit_axis(self):
        source='''@fhe.program(slots=1024)
def f(x: Cipher[1,3,2]):
 a=pack(x,layout="token")
 y=bias(a,b=W["offset"],axis=-1)
 return y
'''
        self.assertTrue(check(parse(source),{'offset':(2,)},prompt_version='r8-v8').ok)
        self.assertIn('bias(x,b=W[...],axis=-1)',prompts.build_system_prompt(context()))


class DomainTests(unittest.TestCase):
    def test_analytic_contains_required_domain_and_narrow_is_calibrated(self):
        from pt_agent.transformer_validation import points
        source='''@fhe.program(slots=1024)
def f(x: Cipher[1,1,2]):
 a=pack(x,layout="token")
 y=softmax(a,steps=2,exp_method="cheb",exp_deg=7,exp_domain=(-1.,0.),inv_method="remez",inv_deg=15,inv_domain=(0.01,1.),inv_scale=2.,step_inv_method="remez",step_inv_deg=15,step_inv_domain=(.1875,2.34375),final_inv_method="remez",final_inv_deg=15,final_inv_domain=(.1875,2.34375),inv_budget=.25)
 return y
'''
        ctx=context(); program=parse(source)
        desc=points(program); sources=criteria.provenance(program,ctx,desc)
        self.assertEqual([s['domain_source'] for s in sources],['calibrated','analytic','analytic','analytic'])
        self.assertEqual(sources[1]['analytic_domain'],(np.exp(-1.)/2,1.))
        self.assertEqual(sources[2]['analytic_domain'],(.1875,2.34375))
        program.ops[-1].attrs['step_inv_domain']=(.2,2.34375)
        sources=criteria.provenance(program,ctx,points(program))
        self.assertEqual(sources[2]['domain_source'],'calibrated')
        self.assertEqual(sources[3]['domain_source'],'analytic')

    def test_legacy_repeated_bound_and_missing_budget(self):
        from types import SimpleNamespace
        op=SimpleNamespace(kind='softmax',attrs=dict(steps=3))
        point=SimpleNamespace(op=op,prefix='step_inv')
        self.assertEqual(criteria.analytic_domain(point,(1,2,40,40)),(1/60,1.5))
        op.attrs['final_inv_method']='remez'
        self.assertIsNone(criteria.analytic_domain(point,(1,2,40,40)))

    def test_shared_u_decision_and_stop(self):
        result=criteria.summarize([dict(domain_source='analytic',max_u=.99),
                                   dict(domain_source='calibrated',max_u=.89)])
        self.assertTrue(criteria.domain_ok(result))
        result['max_u_calibrated']=.91; self.assertFalse(criteria.domain_ok(result))
        result=criteria.summarize([dict(domain_source='analytic',max_u=1.0000001)])
        self.assertFalse(criteria.domain_ok(result))
        with self.assertRaisesRegex(RuntimeError,'§6.7'): criteria.require_analytic_coverage(result)

    def test_cnn_and_old_version_decisions_remain(self):
        for version in ('r8-v4','r8-v5','r8-v6','r8-v7'):
            ctx=context();ctx.prompt_version=version
            self.assertFalse(criteria.enabled(ctx))
        result=criteria.summarize([dict(domain_source='calibrated',max_u=.91)])
        self.assertEqual(result['max_u_analytic'],0.)
        self.assertFalse(criteria.domain_ok(result))

    def test_actual_cnn_validation_retains_all_frozen_fields(self):
        ctx=context()
        source=SOURCE.replace('Cipher[1,1,2]','Cipher[1,2]').replace('layout="token"','layout="flat"')
        program=parse(source)
        ctx.val_images=ctx.val_images.reshape(-1,2)
        ctx.val_inputs={'x':ctx.val_images}
        ctx.model_info.inputs[0]['shape']=[1,2]
        old=[]
        for version in ('r8-v4','r8-v5','r8-v6'):
            ctx.prompt_version=version
            old.append(agent_tools.validate(program,ctx,engine='tensor',cache=False))
        ctx.prompt_version='r8-v8'
        current=agent_tools.validate(program,ctx,engine='tensor',cache=False)
        for result in old:
            for key in ('feasible','agree','total','max_u','diverged','n_diverged','agreement_pct'):
                self.assertEqual(current[key],result[key])
        self.assertEqual(current['max_u_analytic'],0.)
        self.assertEqual(current['max_u_calibrated'],current['max_u'])


@require(COMPILER)
class SessionTests(unittest.TestCase):
    def test_api_refusal_is_counted_and_no_token_fallback(self):
        from pt_agent.llm import LLMError
        info,calibration,samples=fixture()
        raw=RecordedClient([LLMError('http_error',dict(status=400))])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            client=ModelBudgetClient(raw,root/'calls',model='demo',prompt_version='r8-v8')
            result=run_session(info,calibration,samples,client,root/'session',max_iters=8,prompt_version='r8-v8')
            self.assertEqual(result['status'],'error')
            self.assertEqual(result['error'],'http_error')
            self.assertEqual(client.calls,1);self.assertEqual(len(raw.messages),1)
            self.assertTrue((client.directory/'call-001.error.json').exists())
    def test_length_feedback_and_versioned_budget(self):
        class Client(RecordedClient):
            def chat(self,messages,*,purpose):
                reply=super().chat(messages,purpose=purpose)
                reply['finish_reason']='length' if not reply['content'] else 'stop'
                return reply
        info,calibration,samples=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=Client(['',json.dumps(dict(program=SOURCE))])
            client=ModelBudgetClient(raw,root/'calls',model='demo',prompt_version='r8-v8')
            result=run_session(info,calibration,samples,client,root/'session',max_iters=2,
                               tuner=False,prompt_version='r8-v8')
            self.assertEqual(result['status'],'feasible')
            self.assertEqual(result['history'][0]['stage'],'length')
            self.assertIn('65536',raw.messages[1][-1]['content'])
            self.assertEqual(client.calls,2)
            self.assertEqual(ModelBudgetClient(raw,root/'calls',model='demo',prompt_version='r8-v8').calls,2)
            self.assertEqual(ModelBudgetClient(raw,root/'calls').calls,0)
            self.assertEqual(ModelBudgetClient(raw,root/'calls',model='other',prompt_version='r8-v8').calls,0)
            self.assertEqual(OUTPUT_TOKENS,{'r8-v7':32768,'r8-v8':65536})

    def test_confirm_uses_split_u_decision(self):
        ctx=context();ctx.val_split='confirm'
        result=dict(agreement_pct=100.,max_u=.99,max_u_calibrated=.85,max_u_analytic=.99,
                    diverged=False,n_diverged=0,feasible=True,analytic_domain_violation=False,
                    boot_input_bound=None,boot_input_max_abs=None,agree=6,total=6,worst_acts=[])
        with patch.object(confirm,'validate',return_value=result):
            selected=confirm.confirm_select([confirm.Candidate(SOURCE,'llm',1,1,0)],ctx,engine_profile='tensor')
        self.assertIsNotNone(selected['selected'])
        self.assertEqual(selected['evals'][0]['max_u_calibrated'],.85)

    def test_tensor_tuner_finishes_with_mock(self):
        ctx=context()
        from pt_agent.transformer_validation import validate
        with patch.object(transformer_tuner,'validate',wraps=validate) as observed:
            result=transformer_tuner.tune(parse(SOURCE),ctx,budget_s=0,max_sweeps=0,engine_profile='tensor')
        self.assertTrue(result.feasible)
        self.assertEqual(observed.call_args.kwargs['engine'],'mock')
        self.assertIn('max_u_calibrated',result.history[-1])


PREPARED = (RESULTS_DIR/'A1-bert-prepare-20261027T123333-7967',
            RESULTS_DIR/'A1-qwen-prepare-20261027T132722-12c2')


@require(*(directory/'model/model_info.json' for directory in PREPARED))
class RetainedModelTests(unittest.TestCase):
    def test_v7_prompt_and_summary_reproduce_both_first_requests(self):
        from pt_agent.models.info import ModelInfo
        from pt_agent.experiments.a1_transformer_agent import samples_for
        for directory in PREPARED:
            info=ModelInfo.load(directory/'model')
            calibration=json.loads((directory/'model/calibration.json').read_text())
            ctx=agent_tools.build_context(info,calibration,samples=samples_for(directory/'model',info))
            ctx.prompt_version='r8-v7'
            saved=json.loads((directory/'first-request.pretty.json').read_text())['messages']
            self.assertEqual(prompts.build_system_prompt(ctx),saved[0]['content'])
            self.assertEqual(prompts.initial_message(ctx),saved[1]['content'])

    def test_four_baseline_signatures_and_target_names(self):
        from pt_agent.models.info import ModelInfo
        from types import SimpleNamespace
        for directory in PREPARED:
            info=ModelInfo.load(directory/'model')
            for baseline in ('R0','R1'):
                program=parse((directory/baseline/'program.ptdsl').read_text())
                self.assertTrue(check_interface(program,info)['ok'])
            # Prompt rendering needs only sample count, never any sample values.
            ctx=SimpleNamespace(graph=info.graph,model_name=info.graph.name,val_images=range(20),
                                val_labels=None,model_info=info,prompt_version='r8-v8')
            text=prompts.build_system_prompt(ctx)
            for name in info.graph.weights:self.assertNotIn(name,text)


if __name__=='__main__':unittest.main()
