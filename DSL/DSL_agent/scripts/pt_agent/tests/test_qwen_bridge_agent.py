"""Synthetic bridge/Agent protocol tests; no model, input text or API calls."""
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_bridge_agent as worker


class NamespaceCases(unittest.TestCase):
    def test_private_delegation_preserves_default_module(self):
        old = worker.original.MODULE, worker.original._calibrate, worker.original._agent_phase
        namespace = worker._namespace()
        self.assertEqual(namespace['MODULE'], worker.MODULE)
        self.assertIs(namespace['_calibrate'], worker._calibrate)
        self.assertIs(namespace['worker'].__globals__, namespace)
        self.assertIs(namespace['phase_command'].__globals__, namespace)
        self.assertEqual(old, (worker.original.MODULE, worker.original._calibrate, worker.original._agent_phase))

    def test_local_namespace_helpers_use_rebound_dependencies(self):
        module = ModuleType('fixture')
        exec('VALUE=2\ndef inner(): return VALUE\ndef outer(): return inner()\n', module.__dict__)
        namespace = worker._clone_namespace(module, {'VALUE': 7})
        self.assertEqual(namespace['outer'](), 7)
        self.assertEqual(module.outer(), 2)

    def test_source_metadata_identifies_new_budget_and_no_tuner(self):
        with patch.object(worker.original, 'source_gate', return_value={'original': True}):
            record = worker.source_gate(SimpleNamespace(out=None))
        self.assertEqual(record['maximum_square_steps'], 11)
        self.assertEqual(record['maximum_newton_iterations'], 7)
        self.assertFalse(record['tuner_enabled'])
        self.assertFalse(record['raw_inverse_statistics_used'])
        self.assertEqual(record['planner_engine_protocol'], 'r8-v11')


@dataclass
class TinyContext:
    graph: object
    samples: object
    val_images: object
    val_inputs: object
    val_labels: object
    val_exact_top1: object
    val_exact_logits: object
    val_split: str


class BridgeNumericalCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Existing numeric interpreter only. Tests instantiate no trained model.
        import numpy as np
        cls.np = np

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def graph(self, count=1):
        tensors = {'attention_mask': SimpleNamespace(shape=(1,3), kind='mask')}
        nodes = []
        for index in range(count):
            score, out = 'score'+str(index), 'p'+str(index)
            tensors[score] = SimpleNamespace(shape=(1,2,3,3), kind='act')
            tensors[out] = SimpleNamespace(shape=(1,2,3,3), kind='act')
            nodes.append(SimpleNamespace(id='s'+str(index), op='softmax',
                inputs=[score,'attention_mask'], outputs=[out], attrs=dict(axis=-1,keep='causal',mask='attention_mask')))
        return SimpleNamespace(name='synthetic_attention', nodes=nodes, tensors=tensors,
                               inputs=['score0','attention_mask'], outputs=['p0'], weights={})

    def calibration(self, graph):
        return {node.outputs[0]+'::exp': dict(min=-10., max=1010.) for node in graph.nodes}

    def bridge(self, graph):
        from pt_agent.planners.precision_variants import build_unit_width_bridge
        record = build_unit_width_bridge(graph, self.calibration(graph))
        record.update(agent_variant=worker.VARIANT, actual_score_coverage_measured=True,
            stable_float64_math_passed=True, real_forward_states=23,
            source=dict(model_info=str(self.root/'info'), inputs=str(self.root/'inputs'), calibration_indices=list(range(23))))
        return record

    def program(self, policy, iterations=7):
        attrs = {key: policy[key] for key in ('shift','steps','inv_scale')}
        attrs.update(exp_method='cheb', exp_domain=policy['exp_domain'], exp_deg=3,
            inv_method='newton', inv_domain=policy['inv_domain'], inv_deg0=3, inv_iters=iterations,
            step_inv_method='newton', step_inv_domain=policy['step_inv_domain'], step_inv_deg0=3,step_inv_iters=iterations)
        op = SimpleNamespace(id=0,kind='softmax',attrs=attrs,outputs=['p0'],inputs=['x'],lineno=3,call_lines=())
        return SimpleNamespace(ops=[op])

    def test_raw_overflow_remains_diagnostic_without_finite_inverse_substitute(self):
        np = self.np; graph = self.graph(); stats = {}; issues = []
        x = np.array([[[[-10.,0.,1010.],[-10.,0.,1010.],[-10.,0.,1010.]]]*2],dtype=np.float64)
        values = dict(score0=x,p0=np.zeros_like(x),attention_mask=np.ones((1,3),dtype=np.float64))
        worker.collect_points(graph,values,stats,issues,sample_id='synthetic:0')
        self.assertNotIn('p0::inv',stats)
        self.assertEqual((stats['p0::exp']['min'],stats['p0::exp']['max']),(-10.,1010.))
        self.assertFalse(issues[0]['all_finite'])
        self.assertFalse(issues[0]['used_for_fit_or_domain'])
        self.assertTrue(all(np.isfinite(value) for row in stats.values() for value in row.values()))

    def test_new_bridge_binds_training_indices_and_actual_sources(self):
        graph=self.graph(); measured=self.bridge(graph)
        info=SimpleNamespace(graph=graph,split=dict(calibration=list(range(23))))
        args=SimpleNamespace(model_info=self.root/'info',inputs=self.root/'inputs')
        self.assertIs(worker.checked_bridge(info,self.calibration(graph),measured,args),measured)
        measured['source']['inputs']=str(self.root/'wrong-inputs')
        with self.assertRaisesRegex(ValueError,'same newly'):worker.checked_bridge(info,self.calibration(graph),measured,args)

    def test_bridge_rejects_unmeasured_and_modified_parameters(self):
        graph=self.graph(); info=SimpleNamespace(graph=graph,split=dict(calibration=list(range(23))))
        args=SimpleNamespace(model_info=self.root/'info',inputs=self.root/'inputs')
        measured=self.bridge(graph); measured['actual_score_coverage_measured']=False
        with self.assertRaises(ValueError):worker.checked_bridge(info,self.calibration(graph),measured,args)
        measured=self.bridge(graph);calibration=self.calibration(graph);calibration['p0::inv']={'min':1.,'max':2.}
        with self.assertRaisesRegex(ValueError,'Raw inverse'):worker.checked_bridge(info,calibration,measured,args)
        measured=self.bridge(graph); measured['rows'][0]['policy']['steps']=10
        with self.assertRaises(ValueError):worker.checked_bridge(info,self.calibration(graph),measured,args)

    def prompt_context(self):
        graph=self.graph(); graph.inputs=['score0','attention_mask']
        ctx=SimpleNamespace(graph=graph,calibration=self.calibration(graph),profile='qwen48',
            model_name='synthetic_attention',prompt_version='r8-v11',needed_margin={},
            stable_bridge=self.bridge(graph), val_images=self.np.zeros((20,1)), val_labels=None,
            model_info=SimpleNamespace(task='causal_lm',metric='lm_next_token',threshold=dict(agreement_pct=99.,logit_rel_err=.01)))
        return ctx

    def test_prompt_has_truthful11_7_no_tuner_and_no_raw_inverse_numbers(self):
        from pt_agent.planners import prompts
        from pt_agent.dsl import interface
        ctx=self.prompt_context()
        with patch.object(interface,'inputs',return_value=[dict(name='x',type='Cipher[1,3,3]')]):
            message=json.loads(worker.prompt_initial(ctx))
            before=prompts.build_system_prompt(ctx)
            system=worker.prompt_system(ctx)
            after=prompts.build_system_prompt(ctx)
        inverse=message['nodes'][0]['approximation_points'][1]['calib']
        self.assertEqual(inverse['status'],'raw_statistic_unused')
        self.assertNotIn('min',inverse);self.assertNotIn('max',inverse)
        self.assertEqual(message['prompt_variant_id'],worker.PROMPT_ID)
        self.assertFalse(message['tuner_enabled'])
        self.assertIn('steps is0..11',system);self.assertIn('iterations0..7',system)
        self.assertNotIn('steps is an integer in 0..8',system)
        self.assertNotIn('Newton iterations <=5',system)
        self.assertNotIn('After each feasible program, a deterministic tuner',system)
        self.assertTrue('last real position is at least 99%' in system, 'Keep original agreement policy')
        self.assertTrue('at most 1e-2 (logit_rel_err)' in system, 'Keep original relative error policy')
        self.assertEqual(before,after)

    def test_candidate_locks_public_decomposition_and_allows_only_newton7(self):
        bridge=self.bridge(self.graph()); ctx=SimpleNamespace(stable_bridge=bridge)
        program=self.program(bridge['rows'][0]['policy'])
        worker.candidate_gate(program,ctx)
        for key in ('shift','steps','inv_scale'):
            bad=copy.deepcopy(program);bad.ops[0].attrs[key]+=1
            with self.assertRaisesRegex(ValueError,'Preserve'):worker.candidate_gate(bad,ctx)
        for iterations in (8,None,7.,True):
            with self.assertRaisesRegex(ValueError,'iterations'):
                worker.candidate_gate(self.program(bridge['rows'][0]['policy'],iterations),ctx)

    def test_strict_math_checks_dtype_shape_finite_and_original_tolerance(self):
        np=self.np; ref=np.array([0.,1.],dtype=np.float64)
        self.assertTrue(worker.strict_math(ref,ref+1e-13)['ok'])
        self.assertFalse(worker.strict_math(ref,ref+1e-9)['ok'])
        for invalid in (ref.astype(np.float32),ref.reshape(1,2),np.array([np.inf,1.])):
            with self.assertRaises(ValueError):worker.strict_math(ref,invalid)

    def test_small_real_count_and_exact_tensor_preserve_stable_kernel_arithmetic(self):
        np=self.np
        from pt_agent.planners.precision_variants import unit_width_policy
        from pt_agent.dsl.parser import parse
        from pt_agent.dsl.typecheck import check
        from pt_agent.backends import CountBackend
        from pt_agent.lower import lower
        from pt_agent.tensor_sim import run_tensor
        policy=unit_width_policy(dict(min=-1.,max=1.),3)
        fields=', '.join(key+'='+repr(policy[key]) for key in ('shift','steps','inv_scale'))
        for role in ('exp','inv','step_inv'):
            fields+=(', '+role+'_method="cheb", '+role+'_deg=3' if role=='exp' else
                     ', '+role+'_method="newton", '+role+'_deg0=3, '+role+'_iters=7')
            fields+=', '+role+'_domain='+repr(tuple(policy[role+'_domain']))
        source=('@fhe.program(slots=16384, refresh="compiler", placement="depth-dp")\n'
            'def tiny(x: Cipher[1,2,3,3], mask: Mask[1,3]):\n'
            '    s=pack(x,layout="score")\n    m=pack(mask,layout="flat")\n'
            '    p=softmax(s,axis=-1,keep="causal",mask=m, '+fields+')\n    return p\n')
        program=parse(source)
        checked=check(program,{},prompt_version='r8-v11')
        self.assertTrue(checked.ok,repr(checked.diags))
        worker.candidate_gate(program,SimpleNamespace(stable_bridge={'rows':[{'policy':policy}]}))
        backend=CountBackend(slots=16384)
        values,layouts=lower(program,{},backend)
        self.assertGreater(backend.stats()['ops_total'],0)
        self.assertEqual(len(values),3)  # One original ScoreLayout ciphertext per query.
        self.assertEqual(len(layouts),1)  # One logical output.
        x=np.linspace(-1.,1.,18,dtype=np.float64).reshape(1,2,3,3)
        mask=np.ones((1,3),dtype=np.float64)
        keep=np.arange(3)[None,:]<=np.arange(3)[:,None]
        masked=np.where(keep,x,-np.inf);e=np.exp(masked-masked.max(-1,keepdims=True))
        independent=e/e.sum(-1,keepdims=True)
        observed=run_tensor(program,{},dict(x=x,mask=mask),act_mode='exact',cache=False)
        self.assertTrue(worker.strict_math(independent,observed)['ok'])

    def test_confirmation_and_c2_keep_framework_quality_and_original_partitions(self):
        np=self.np; accesses=[]; graph=SimpleNamespace(inputs=['x'],outputs=['y'])
        class Samples:
            def get(self,name):
                accesses.append(name);return {'x':np.ones(({'calibration':23,'heldout':20,'confirm':20}[name],1))},None
        class Reference:
            def forward(self,arrays,*,framework,keep=False):
                if not framework:raise AssertionError('Confirmation may not replace its quality reference')
                return {'y':np.full((1,1),7.,dtype=np.float64)}
        ctx=TinyContext(graph,Samples(),None,None,None,None,None,'heldout')
        ctx.stable_bridge={'rows':[]};ctx.reference_owner=Reference()
        ctx.prompt_variant_id=worker.PROMPT_ID;ctx.algorithm_variant=worker.VARIANT
        confirm=worker.partition_context(ctx,splits=('confirm',))
        self.assertEqual(accesses,['confirm']);self.assertEqual(confirm.val_exact_logits.shape,(20,1))
        self.assertTrue((confirm.val_exact_logits==7).all());self.assertIs(confirm.stable_bridge,ctx.stable_bridge)
        self.assertIsNone(confirm.math_exact_logits)
        accesses.clear();tail=worker.partition_context(ctx,splits=('calibration','heldout','confirm'))
        self.assertEqual(accesses,['calibration','heldout','confirm'])
        self.assertEqual(len(tail.tail_sample_positions),63)
        with self.assertRaises(ValueError):worker.partition_context(ctx,splits=('report',))

    def test_exact_math_has_canonical_reference_and_quality_context_unchanged(self):
        np=self.np;from pt_agent import transformer_validation as validation
        bridge=self.bridge(self.graph());program=self.program(bridge['rows'][0]['policy'])
        ctx=SimpleNamespace(stable_bridge=bridge,math_exact_logits=np.array([[2.]],dtype=np.float64),
            val_exact_logits=np.array([[7.]],dtype=np.float64),val_inputs={'x':np.ones((1,1))},weights={})
        def old(program,proxy,**kw):
            self.assertTrue((proxy.val_exact_logits==2).all());return {'ok':True}
        with patch.object(validation,'exact_check',side_effect=old),patch.object(validation,'execute',return_value=np.array([[2.]])):
            result=worker.exact_check(program,ctx)
        self.assertTrue(result['ok']);self.assertTrue(result['quality_reference_separate'])
        self.assertTrue((ctx.val_exact_logits==7).all())

    def test_agent_private_namespace_leaves_original_evaluator_and_confirmation_untouched(self):
        from pt_agent.planners import agent,confirm
        before=agent.evaluate,agent.confirm_select,agent.initial_message,confirm.build_confirm_context
        namespace=worker._agent_functions()
        self.assertIs(namespace['agent_step'].__globals__,namespace)
        self.assertIsNot(namespace['evaluate'],agent.evaluate)
        self.assertIsNot(namespace['confirm_select'],agent.confirm_select)
        self.assertEqual(before,(agent.evaluate,agent.confirm_select,agent.initial_message,confirm.build_confirm_context))

    def test_calibration23_mapped_prefixes_and552_actual_small_stable_checks(self):
        np=self.np;graph=self.graph(24)
        info=SimpleNamespace(graph=graph,split=dict(calibration=list(range(23)),sample_ids=['synthetic:'+str(i) for i in range(23)]))
        class Samples:
            def get(self,name):
                if name!='calibration':raise AssertionError('Training calibration only')
                return dict(score0=np.zeros((23,2,3,3),dtype=np.float64),attention_mask=np.ones((23,3),dtype=np.float64)),None
        class Reference:
            def __init__(self,info):
                self.evidence=['synthetic']; self.framework=self.canonical=SimpleNamespace(observation=dict(private_weight_copies=0))
                self.counts={}
            def forward(self,arrays,*,framework,keep):
                values={'attention_mask':arrays['attention_mask']}
                x=np.array([[[[-10.,0.,1010.],[-10.,0.,1010.],[-10.,0.,1010.]]]*2],dtype=np.float64)
                for node in graph.nodes:
                    values[node.inputs[0]]=x;values[node.outputs[0]]=np.zeros_like(x)
                return values
        class Recorder:
            def span(self,*args,**kwargs):
                from contextlib import nullcontext
                return nullcontext()
        work=self.root/'work';work.mkdir()
        args=SimpleNamespace(work_dir=work,calibration=self.root/'calibration.json',model_info=self.root/'info',inputs=self.root/'inputs')
        result={}
        write=worker.write_json
        def small_checkpoint(path,value):
            # Keep the actual552 checks and all final artifacts. Another test
            # exercises real SESSION/prompt persistence; avoid serializing the
            # cumulative synthetic check list552 times in this numerical test.
            if Path(path).name!='worker-result.json':write(path,value)
        with patch.object(worker.original,'_load_model',return_value=(info,Samples())),patch.object(worker,'References',Reference),patch.object(worker,'write_json',side_effect=small_checkpoint):
            worker._calibrate(args,Recorder(),result)
        self.assertEqual((result['calibration_forward_attempts'],result['calibration_forward_completed']),(23,23))
        self.assertEqual((result['stable_softmax_attempts'],result['stable_softmax_completed']),(552,552))
        self.assertTrue(all(row['passed'] for row in result['stable_softmax_checks']))
        saved=json.loads(args.calibration.read_text())
        self.assertFalse(any(name.endswith('::inv') for name in saved))
        self.assertEqual(len(result['raw_inverse_diagnostics']),552)
        self.assertTrue(result['bridge_calibration_completed'])

    def test_prompt_only_persists_no_tuner_state_and_never_creates_client(self):
        from contextlib import nullcontext
        args=SimpleNamespace(session=self.root/'session',work_dir=self.root,worker_phase='prompt')
        state=dict(phase='evaluate',iteration=1,calls=0,max_iters=10,prompt_version='r8-v11',placement='program-plan',
            confirm_rule='c2',tuner=False,messages=[dict(role='system',content='Synthetic DSL instructions'),dict(role='user',content='Synthetic metadata')])
        def new(context,**kwargs):
            self.assertFalse(kwargs['tuner']);return copy.deepcopy(state)
        functions=dict(new_agent_state=new,agent_step=lambda *a,**k:self.fail('No API in prompt phase'))
        recorder=SimpleNamespace(span=lambda *a,**k:nullcontext())
        with patch.object(worker,'_context',return_value=object()),patch.object(worker,'_agent_functions',return_value=functions):
            result={};worker._agent_phase(args,recorder,result)
        self.assertTrue(result['prompt_completed'])
        saved=json.loads((args.session/'SESSION.json').read_text())
        self.assertFalse(saved['tuner']);self.assertEqual(saved['prompt_variant_id'],worker.PROMPT_ID)
        saved['tuner']=True;(args.session/'SESSION.json').write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError,'no-tuner'):worker.session_outcome(args.session)


if __name__=='__main__':unittest.main()
