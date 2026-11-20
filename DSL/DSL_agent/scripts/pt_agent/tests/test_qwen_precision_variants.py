"""Explicit precision variants on deterministic small arithmetic and one fit."""

import copy
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent.planners import stable_softmax_bridge as old
from pt_agent.planners import precision_variants as variant


EVIDENCE = {}


def graph():
    return SimpleNamespace(name='attention',nodes=[SimpleNamespace(id='s',op='softmax',
        inputs=['x','mask'],outputs=['p'],attrs=dict(keep='causal',mask='mask'))],
        tensors={'x':SimpleNamespace(shape=(1,2,3,3))})


def report():
    profile = dict(depth=7,minimum_initial_z=.4,maximum_initial_z=1.6,
        diagnostic_points=17,newton_model=dict(history=[.6,.5,.4,.3,.2,.05,.003,.0001]))
    return dict(fn='rsqrt',domain=[.1,100.],target=.001,status='infeasible',selected=None,
        selected_depth=None,candidates=[dict(method='newton',fit='relative',deg0=31,iters=5,
        depth=22,relative_error=.05,feasible=False,profile=profile)])


class PrecisionVariantProtocolCases(unittest.TestCase):
    def test_explicit_k11_policy_preserves_original_cap8(self):
        observations=dict(min=-10.534944156119677,max=1010.9114519970741)
        before=old.derive_policy(observations,32)
        policy=variant.unit_width_policy(observations,32)
        self.assertEqual((before['steps'],policy['steps']),(8,11))
        self.assertLessEqual(policy['actual_initial_exponent_width'],1.)
        self.assertLess(policy['inv_domain'][1]/policy['inv_domain'][0],32*math.e*1.001)
        self.assertEqual(old.derive_policy(observations,32),before)
        with self.assertRaisesRegex(ValueError,'k<=11'):
            variant.unit_width_policy(dict(min=-2000.,max=2000.),32)

    def test_json_source_match_and_planner_restore_for_named_variant(self):
        calibration={'p::exp':dict(min=-10.,max=1010.)}
        measured=variant.build_unit_width_bridge(graph(),calibration)
        measured['actual_score_coverage_measured']=True
        measured=json.loads(json.dumps(measured))
        original=old.transformer_rule.squared_softmax_adjustments

        def planned(*args,**kwargs):
            self.assertEqual(old.transformer_rule.squared_softmax_adjustments(calibration['p::exp'],3)['steps'],11)
            return 'unfrozen explicit variant'

        with mock.patch.object(old.transformer_rule,'plan',side_effect=planned) as planner:
            self.assertEqual(old.plan_with_bridge(graph(),calibration,measured,
                policy_builder=variant.build_unit_width_bridge),'unfrozen explicit variant')
            measured['rows'][0]['policy']['steps']=10
            with self.assertRaises(ValueError):
                old.plan_with_bridge(graph(),calibration,measured,policy_builder=variant.build_unit_width_bridge)
            self.assertEqual(planner.call_count,1)
        self.assertIs(old.transformer_rule.squared_softmax_adjustments,original)

    def test_persisted_domain_lists_roundtrip_through_real_planner_and_parser(self):
        from pt_agent.dsl.parser import parse
        small=graph()
        small.inputs=['x','mask'];small.outputs=['p']
        small.tensors['mask']=SimpleNamespace(shape=(1,3),kind='mask')
        small.tensors['x'].kind='cipher'
        calibration={'p::exp':dict(min=-10.,max=1010.)}
        measured=variant.build_unit_width_bridge(small,calibration)
        measured['actual_score_coverage_measured']=True
        measured=json.loads(json.dumps(measured))
        self.assertIsInstance(measured['rows'][0]['policy']['exp_domain'],list)
        def recorded(fn,domain,**kwargs):
            return dict(selected=dict(method='remez',deg=7,fit='relative'))
        text=old.plan_with_bridge(small,calibration,measured,
            policy_builder=variant.build_unit_width_bridge,approximation_selector=recorded)
        program=parse(text)
        softmax=next(op for op in program.ops if op.kind=='softmax')
        for key in ('exp_domain','inv_domain','step_inv_domain'):
            self.assertEqual(softmax.attrs[key],tuple(measured['rows'][0]['policy'][key]))
        self.assertEqual(softmax.attrs['steps'],11)

    def test_newton7_budget_reuses_evidence_without_changing_original(self):
        original=report();before=copy.deepcopy(original)
        result=variant.extend_newton_budget(original)
        self.assertEqual(original,before)
        self.assertEqual(result['selected'],dict(method='newton',deg0=31,iters=7,fit='relative'))
        self.assertEqual(result['selected_depth'],28)
        self.assertEqual(result['additional_candidates'],2)
        self.assertFalse(result['continuous_error_certificate'])
        reused=variant.extend_newton_budget(result)
        self.assertEqual(reused['selected'],result['selected'])
        self.assertEqual(reused['candidates'],result['candidates'])
        self.assertEqual(reused['additional_candidates'],0)
        outside=report();outside['candidates'][0]['profile']['maximum_initial_z']=2.
        self.assertIsNone(variant.extend_newton_budget(outside)['selected'])

    def test_recorded_requests_match_domains_target_and_function(self):
        original=report()
        records=[dict(kind='end',fn='rsqrt',domain=[.1,100.],settings=dict(target=.001),report=original)]
        def fresh(fn,domain,**kwargs):
            return dict(original,fn=fn,domain=list(domain),target=kwargs.get('target',.001))
        fallback=mock.Mock(side_effect=fresh)
        select,statistics=variant.recorded_selector(records,fallback,newton_budget=7)
        self.assertEqual(select('rsqrt',(.1,100.),target=.001)['selected']['iters'],7)
        fallback.assert_not_called()
        self.assertEqual(statistics['reused_reports'],1)
        select('rsqrt',(.2,100.),target=.001)
        fallback.assert_called_once()
        self.assertEqual(statistics['fresh_fit_calls'],1)
        changed=copy.deepcopy(records);changed[0]['report']['domain']=[.2,100.]
        with self.assertRaises(ValueError):
            variant.recorded_selector(changed,fallback,newton_budget=7)
        different=copy.deepcopy(records[0]);different['report']['candidates'][0]['relative_error']=.06
        with self.assertRaises(ValueError):
            variant.recorded_selector(records+[different],fallback,newton_budget=7)

    def test_new_jobs_expose_separate_budgets_and_source_reports(self):
        from pt_agent.experiments import qwen_diagnostic_jobs as jobs
        args=SimpleNamespace(job='stable-calibration',out=Path('/synthetic/out'),
            linear_variant='compact-rectangular',decomposition_variant=variant.SQUARE_VARIANT,newton_budget=7,
            model_info=Path('/synthetic/model'),original_hf_reference=Path('/synthetic/hf.json'),
            framework_result=Path('/synthetic/framework.json'),stable_calibration_result=Path('/synthetic/bridge.json'),
            fit_source=Path('/synthetic/original-fit.json'))
        calibration=jobs.job_definition(args)
        self.assertEqual((calibration['memory_gib'],calibration['timeout']),(6,960))
        self.assertNotIn('--fit-source',calibration['command'])
        args.job='stable-count'
        count=jobs.job_definition(args)
        self.assertEqual((count['memory_gib'],count['timeout']),(12,1860))
        self.assertIn('--fit-source',count['command'])
        self.assertEqual(count['command'][count['command'].index('--newton-budget')+1],'7')
        self.assertIn(variant.SQUARE_VARIANT,count['mode'])
        args.job='stable-fit'
        fit=jobs.job_definition(args)
        self.assertNotIn('pt_agent.tools.nixrun',fit['command'])
        self.assertEqual((fit['memory_gib'],fit['timeout'],fit['resident_limit_mib']),(6,900,600))
        self.assertEqual(fit['command'][fit['command'].index('--stop-after')+1],'fit')

    def test_server_preserves_explicit_precision_and_prior_fitting_source(self):
        from pt_agent.experiments import qwen_server_job as server
        from pt_agent.experiments import qwen_diagnostic_jobs as jobs
        for stage in ('stable-calibration','stable-fit','stable-count'):
            with self.subTest(stage=stage),mock.patch.object(jobs,'main',return_value=0) as dispatched:
                self.assertEqual(server.main(['--out','/synthetic/out','--stage',stage,
                    '--decomposition-variant',variant.SQUARE_VARIANT,'--newton-budget','7',
                    '--fit-source','/synthetic/old-fit.json']),0)
            forwarded=dispatched.call_args.args[0]
            self.assertEqual(forwarded[forwarded.index('--job')+1],stage)
            self.assertEqual(forwarded[forwarded.index('--decomposition-variant')+1],variant.SQUARE_VARIANT)
            self.assertEqual(forwarded[forwarded.index('--newton-budget')+1],'7')
            self.assertEqual(forwarded[forwarded.index('--fit-source')+1],'/synthetic/old-fit.json')

    def test_fit_guard_rejects_job_residency_and_host_memory_reserve(self):
        from pt_agent.experiments import qwen_stable_calibration as worker
        for snapshot,message in (((600*1024+1,3*1024**2),'job RSS'),
                                 ((20*1024,2*1024**2-1),'Host available')):
            observation={}
            with self.subTest(snapshot=snapshot), \
                    mock.patch.object(worker,'_fit_resident_snapshot',return_value=snapshot), \
                    mock.patch.object(worker.signal,'getitimer',return_value=(0.,0.)), \
                    mock.patch.object(worker.signal,'getsignal',return_value='original-handler'), \
                    mock.patch.object(worker.signal,'signal') as handler, \
                    mock.patch.object(worker.signal,'setitimer') as timer:
                with self.assertRaisesRegex(MemoryError,message):
                    with worker._fit_resident_guard(observation=observation):
                        self.fail('A rejected isolated job must stop before fitting')
                self.assertTrue(observation['stopped'])
                self.assertEqual(observation['samples'],1)
                timer.assert_called_once_with(worker.signal.ITIMER_REAL,0.)
                self.assertEqual(handler.call_args.args,(worker.signal.SIGALRM,'original-handler'))

    def test_fit_guard_counts_dedicated_supervisor_and_descendants(self):
        from pt_agent.experiments import qwen_stable_calibration as worker
        from pt_agent.experiments import r10_paper_metrics as metrics
        table={10:dict(ppid=1,vmrss_kib=10),11:dict(ppid=10,vmrss_kib=20),
               12:dict(ppid=11,vmrss_kib=30),99:dict(ppid=1,vmrss_kib=10000)}
        with mock.patch.object(metrics,'_scan',return_value=table), \
                mock.patch.object(worker.os,'getpid',return_value=11), \
                mock.patch.object(worker.os,'getppid',return_value=10), \
                mock.patch.object(worker.Path,'read_text',return_value='MemAvailable: 3000000 kB\n'):
            self.assertEqual(worker._fit_resident_snapshot(),(60,3000000))

    def test_fit_metadata_namespace_loads_real_records_and_restores_modules(self):
        import importlib
        from pt_agent.experiments import qwen_stable_calibration as worker
        before={key:value for key,value in sys.modules.items()
                if key=='pt_agent.intake' or key.startswith('pt_agent.intake.')}
        with mock.patch.dict(sys.modules):
            for key in before:
                sys.modules.pop(key)
            original_import=__import__
            def import_without_torch(name,*args,**kwargs):
                if name=='torch' or name.startswith('torch.'):
                    raise AssertionError('Metadata fitting must not initialize a Torch frontend')
                return original_import(name,*args,**kwargs)
            with mock.patch('builtins.__import__',side_effect=import_without_torch):
                with worker._metadata_intake_namespace() as metadata:
                    graph=importlib.import_module('pt_agent.intake.graph')
                    core=importlib.import_module('pt_agent.intake.core_ops')
                    self.assertEqual(Path(graph.__file__).name,'graph.py')
                    self.assertEqual(Path(core.__file__).name,'core_ops.py')
                    self.assertTrue(metadata['package_initializer_skipped'])
                    with tempfile.TemporaryDirectory() as temporary:
                        original=graph.Graph('tiny',[],{}, {'private':graph.np.array([2.])},[],[])
                        original.save(temporary)
                        loaded=graph.Graph.load(temporary)
                    self.assertEqual(loaded.nodes,[])
                    self.assertEqual(loaded.weights['private'].tolist(),[2.])
                    self.assertFalse(loaded.weights['private'].flags.writeable)
                self.assertNotIn('pt_agent.intake',sys.modules)
                self.assertNotIn('pt_agent.intake.graph',sys.modules)
        self.assertEqual({key:value for key,value in sys.modules.items()
            if key=='pt_agent.intake' or key.startswith('pt_agent.intake.')},before)

    def test_fit_metadata_namespace_preserves_an_already_loaded_package(self):
        from pt_agent.experiments import qwen_stable_calibration as worker
        existing=SimpleNamespace(original_package='preserved')
        with mock.patch.dict(sys.modules,{'pt_agent.intake':existing}):
            with worker._metadata_intake_namespace() as metadata:
                self.assertIs(sys.modules['pt_agent.intake'],existing)
                self.assertTrue(metadata['original_package_already_loaded'])
                self.assertFalse(metadata['package_initializer_skipped'])
            self.assertIs(sys.modules['pt_agent.intake'],existing)


class PrecisionVariantMathCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name]='1'
        import numpy as np
        cls.np=np

    def test_original_oracle_stays_cap8_and_new_oracle_covers_k11(self):
        np=self.np
        from pt_agent.kernels.stable_softmax_reference import fixed_shift_squaring as previous
        from pt_agent.kernels.unit_width_softmax_reference import fixed_shift_squaring
        x=np.array([[[1010.,1009.,1008.],[-10.,-9.,-8.],[1.,2.,3.]]],dtype=np.float64)
        keep=np.ones_like(x,dtype=bool)
        with self.assertRaises(ValueError):
            previous(x,keep,shift=1010.,steps=11)
        observed,stages=fixed_shift_squaring(x,keep,shift=1010.,steps=11)
        exponents=np.exp(x-x.max(-1,keepdims=True));expected=exponents/exponents.sum(-1,keepdims=True)
        self.assertTrue(np.all(np.abs(observed-expected)<=1e-12+1e-12*np.abs(expected)))
        self.assertEqual(len(stages['stages']),12)

    def test_original_packed_kernel_accepts_new_explicit_steps_with_strict_math(self):
        np=self.np
        from pt_agent.backends.streaming_mock import StreamingMockBackend
        from pt_agent.kernels.layout import ScoreLayout,FlatLayout,pack_sequence,pack_flat,unpack_sequence
        from pt_agent.kernels.transformer import softmax
        from pt_agent.kernels.unit_width_softmax_reference import fixed_shift_squaring
        measured=variant.build_unit_width_bridge(graph(),{'p::exp':dict(min=-10.,max=1010.)})
        row=measured['rows'][0];policy=row['policy']
        x=np.array([[[[1010.,1009.,1008.],[-10.,-9.,-8.],[1.,2.,3.]],
                     [[-.25,.5,1.],[3.,2.,1.],[.1,.2,.3]]]],dtype=np.float64)
        mask=np.array([[1.,1.,0.]],dtype=np.float64)
        expected,check=old.observe_softmax(x,mask,row,reference=fixed_shift_squaring)
        layout,mask_layout=ScoreLayout(16,3,2,3),FlatLayout(16,3)
        backend=StreamingMockBackend(16)
        backend.bind(pack_sequence(x[0],layout)+[pack_flat(mask[0],mask_layout)])
        inputs=[backend.input(i) for i in range(4)]
        result,out=softmax(backend,inputs[:3],layout,dict(fn='exp',domain=policy['exp_domain']),
            dict(fn='inv',domain=policy['inv_domain']),keep='causal',mask=[inputs[3]],mask_layout=mask_layout,
            shift=policy['shift'],inv_scale=policy['inv_scale'],steps=11,
            step_inv=dict(fn='inv',domain=policy['step_inv_domain']),exact=True)
        actual=unpack_sequence([value.value for value in result],out)[None,...]
        self.assertTrue(check['passed'])
        self.assertTrue(np.all(np.abs(actual-expected)<=1e-12+1e-12*np.abs(expected)))

    def test_worst_recorded_rsqrt_single_fit_quality_and_kernel_arithmetic(self):
        np=self.np
        from pt_agent.planners.transformer_rule import _relative_profile
        from pt_agent.kernels import approx
        from pt_agent.backends import CountBackend
        from pt_agent.backends.streaming_mock import StreamingMockBackend
        domain=(.0648680017130538,3242.6245687005407)
        profile,z=_relative_profile('rsqrt',31,domain,'relative')
        self.assertGreater(float(z.min()),0.)
        self.assertLess(float(z.max()),math.sqrt(3.))
        errors={}
        for iteration in range(1,8):
            z=z*(1.5-.5*z*z)
            errors[iteration]=float(np.abs(1-z).max())
        self.assertGreater(errors[5],.001)
        self.assertGreater(errors[6],.001)
        self.assertLessEqual(errors[7],.001)
        coefficients=approx.remez('rsqrt',31,domain,fit='relative')
        x=np.geomspace(*domain,32)
        independent=np.polynomial.chebyshev.chebval(x*(2/(domain[1]-domain[0]))-
            (domain[1]+domain[0])/(domain[1]-domain[0]),coefficients)
        half=x*.5
        for _ in range(7):
            independent=independent*(1.5-half*(independent*independent))
        backend=StreamingMockBackend(32);backend.bind([x])
        actual=approx.evaluate(backend,backend.input(0),'rsqrt','newton',domain,coefficients,iters=7).value
        difference=np.abs(actual-independent)
        self.assertTrue(np.all(difference<=1e-12+1e-12*np.abs(independent)))
        depths={}
        for iterations in (5,6,7):
            count=CountBackend(32)
            output=approx.evaluate(count,count.input(0),'rsqrt','newton',domain,coefficients,iters=iterations)
            depths[iterations]=output.depth
        self.assertEqual(depths[6]-depths[5],3)
        self.assertEqual(depths[7]-depths[5],6)
        EVIDENCE['worst_rsqrt']=dict(domain=domain,degree=31,fit='relative',target=.001,
            diagnostic_points=profile['diagnostic_points'],initial_z_min=profile['minimum_initial_z'],
            initial_z_max=profile['maximum_initial_z'],errors=errors,depths=depths,
            independent_math_elements=32,max_math_absolute_difference=float(difference.max()),
            math_criterion='1e-12+1e-12*abs(reference)',continuous_error_certificate=False)

    def test_named_calibration_preserves_real_score_cache_and_raw_issues(self):
        from pt_agent.experiments import qwen_stable_calibration as worker
        from pt_agent.experiments import qwen_reference_worker as reference
        np=self.np
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            args=SimpleNamespace(work_dir=root,model_info=root/'model',reference=root/'hf.json',
                framework_result=root/'framework.json',phase='calibration',decomposition_variant=variant.SQUARE_VARIANT)
            source=({},dict(calibration=str(root/'old.json'),calibration_issues=[dict(point='p::inv')]),
                dict(original_hf_source=str(root/'original-hf')),{'p::exp':dict(min=-10.,max=1010.)})
            result=dict(mapped_forward_attempts=0,mapped_forward_completed=0,
                stable_softmax_attempts=0,stable_softmax_completed=0,candidate_frozen=False)
            x=np.zeros((1,2,3,3),dtype=np.float64);x[:,:,0,0]=1010.
            mask=np.array([[1.,1.,0.]],dtype=np.float64)

            def mapped(selected,recorder,output,*modules):
                for index in range(6):
                    output['mapped_forward_attempts']+=1
                    selected._observation_callback(graph(),dict(x=x,mask=mask),
                        dict(sample_id='train:'+str(index//2)),dict(step=index%2),dict(passed=True))
                    output['mapped_forward_completed']+=1
                output['calibration_complete']=False

            recorder=SimpleNamespace(span=lambda *a,**kw:nullcontext())
            with mock.patch.object(worker,'_source',return_value=source), \
                    mock.patch('pt_agent.models.mapped_info.load_mapped_info',return_value=SimpleNamespace(graph=graph())), \
                    mock.patch.object(reference,'mapped_reference',side_effect=mapped):
                worker.calibrate(args,recorder,result,np,None)
            self.assertEqual((result['mapped_forward_completed'],result['stable_softmax_completed']),(6,6))
            self.assertEqual(len(result['score_cache']),6)
            for entry in result['score_cache']:
                self.assertTrue(np.array_equal(np.load(entry['scores'],allow_pickle=False),x))
                self.assertTrue(np.array_equal(np.load(entry['mask'],allow_pickle=False),mask))
            self.assertEqual(result['bridge']['rows'][0]['policy']['steps'],11)
            self.assertFalse(result['raw_calibration_complete'])
            self.assertFalse(result['candidate_frozen'])

    def test_fit_only_stops_after_real_planning_and_typecheck_without_lower(self):
        from pt_agent.experiments import qwen_stable_calibration as worker
        from pt_agent.experiments import qwen_diagnostic_jobs as jobs
        from pt_agent.models import mapped_info
        from pt_agent.planners import stable_softmax_bridge
        np=self.np
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            args=SimpleNamespace(work_dir=root,model_info=root/'model',reference=root/'hf.json',
                framework_result=root/'framework.json',bridge_result=root/'measured.json',
                linear_variant='compact-rectangular',decomposition_variant=old.VARIANT,
                newton_budget=5,fit_source=None,stop_after='fit')
            bridge_path=root/'bridge.json';bridge_path.write_text(json.dumps(dict(variant=old.VARIANT)))
            calibration_path=root/'calibration.json';calibration_path.write_text('{}')
            args.bridge_result.write_text(json.dumps(dict(status='ok',stable_bridge_validated=True,
                selected_decomposition_calibration_complete=True,source=dict(model_info=str(args.model_info),
                original_hf_reference=str(args.reference)),bridge_path=str(bridge_path),calibration=str(calibration_path))))
            info=SimpleNamespace(graph=SimpleNamespace(weights={'private':np.array([2.])}),mapped_weights={'members':1})
            result=dict(rule_plan_attempts=0,rule_plan_completed=0,count_attempts=0,count_completed=0)
            with mock.patch.object(worker,'_source'), \
                    mock.patch.object(mapped_info,'load_mapped_info',return_value=info), \
                    mock.patch.object(stable_softmax_bridge,'plan_with_bridge',return_value='explicit unfrozen program') as plan, \
                    mock.patch('pt_agent.dsl.parser.parse',return_value=SimpleNamespace()), \
                    mock.patch('pt_agent.dsl.typecheck.check',return_value=SimpleNamespace(ok=True)), \
                    mock.patch('pt_agent.lower.lower') as lower:
                worker.count(args,SimpleNamespace(span=lambda *a,**kw:nullcontext()),result)
            plan.assert_called_once()
            lower.assert_not_called()
            jobs.check_completed(result,'stable-fit','compact-rectangular')
            self.assertEqual((result['rule_plan_completed'],result['count_attempts'],result['count_completed']),(1,0,0))
            self.assertEqual(result['weight_payloads_read'],0)


if __name__=='__main__':
    unittest.main()
