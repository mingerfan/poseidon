"""New bounded saved-metadata protocol tests; no model or large Count."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_component_plan_cache as cache
from pt_agent.experiments.r10_paper_metrics import StageRecorder


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False))


def program_and_bindings():
    from pt_agent.dsl.parser import parse
    from pt_agent.experiments import qwen_layered_structure as layered
    text=('def qwen_layer(x, am, w):\n    y=bias(x,b=W[w],axis=-1)\n    out=add(y,x)\n    return out\n'
        '@fhe.program(slots=16384,refresh="agent",placement="program-plan",l_max=5,boot_out_level=5)\n'
        'def qwen2(inputs_embeds: Cipher[1,32,896],attention_mask: Mask[1,32],sel: Mask[1,32]):\n'
        '    x=pack(inputs_embeds,layout="token")\n    am=pack(attention_mask,layout="flat")\n'
        '    sm=pack(sel,layout="flat")\n')
    shapes={};current='x'
    for index in range(24):
        name=f'original.model.layers.{index}.fixture.weight';shapes[name]=(896,)
        text+=f'    h{index}=qwen_layer({current},am,"{name}")\n';current=f'h{index}'
    text+=(f'    n=rmsnorm({current},w=W["original.model.norm.weight"],eps=0.001,'
        'rsqrt_method="newton",rsqrt_domain=(0.01,0.2),rsqrt_deg0=7,rsqrt_iters=2)\n'
        '    selected=select(n,sm)\n    logits=linear(selected,w=W["original.model.embed_tokens.weight"])\n'
        '    return logits\n')
    shapes['original.model.norm.weight']=(896,);shapes['original.model.embed_tokens.weight']=(151936,896)
    partition=layered.split_program(parse(text),shapes)
    return text,partition,[layered.source_binding(s)for s in (*partition.layers,partition.head)]


def fixture(root):
    from pt_agent.planners.transformer_session import save_state,json_safe
    case=(Path(root)/'source').resolve();case.mkdir()
    text,partition,bindings=program_and_bindings()
    session=case/'session';session.mkdir();(session/'program.ptdsl').write_text(text)
    source=dict(source_case=str(case),entry_module=cache.MODULE,original_source=dict(component_cost_policy=cache.POLICY),
        program=str(session/'program.ptdsl'),selected_iteration=13,selected_origin='llm',candidate_frozen=True)
    selected=dict(iteration=13,origin='llm',source=text)
    costroot=session/'component-cost-1';costroot.mkdir();components=[]
    for name,binding in zip(cache.NAMES,bindings):
        boundary=19 if name=='head' else 32
        plan=dict(format='pt-agent-refresh-plan-v1',source='pt_agent.lower',placement='depth-dp',
            calls=35,boots=0,levels=dict(input_level=5,boot_in_level=1,boot_out_level=5),
            substitutions=[],outputs=[],dead=[],merged=[],cuts=[],barriers=0,latency_us=1.,
            timing=dict(lower_s=0.,dp_s=0.,candidates_s=0.,map_s=0.),candidates={'depth-dp':dict(seconds=0.)})
        target=costroot/name
        write(target/'refresh-plan.json',plan);write(target/'source-binding.json',binding)
        count=dict(stats=dict(counts=dict(boot=boundary,add=1),ops_total=boundary+1),
            interface=dict(status='ok',source_binding=binding,explicit_boundary_bootstrap_calls=boundary,
                internal_plan_bootstrap_calls=0,symbolic_output_levels=[5]*boundary))
        write(target/'complete-count.json',count)
        components.append(dict(name=name,plan=str(target/'refresh-plan.json'),internal_boots=0,
            boundary_boots=boundary,output_levels=[5]*boundary))
    cost=dict(placement_model='component-program-plan',lowering_policy=cache.POLICY,
        refresh_plan=dict(components=components),counts=dict(boot=787,add=25),ops_total=812,
        complete_count_replay_counts=dict(boot=787,add=25),complete_count_replay_ops_total=812,
        boots_pred=787,component_internal_boots=0,explicit_boundary_boots=787,max_depth=0,
        rot_keys=0,distinct_consts=0,const_bytes=0,latency_pred_us=25.)
    write(costroot/'result.json',dict(status='ok',**cost));write(costroot/'partial.json',dict(status='running',completed=25,components=list(cache.NAMES)))
    record=dict(cost,iteration=13,source='llm')
    state=dict(phase='done',failed=None,calls=13,pending=None,component_cost_policy=cache.POLICY,
        confirmation=dict(selected=selected),records={(13,'llm',text):(record,1)},best=record,best_program=text)
    save_state(session/'SESSION.json',state)
    write(session/'AGENT.json',dict(confirm=dict(selected=selected),best=json_safe(record),best_program=text))
    write(session/'FROZEN.json',dict(program=str(session/'program.ptdsl'),iteration=13,origin='llm',confirm=dict(selected=selected)))
    write(session/'component-c2-access.json',dict(status='completed',calls=13))
    result=dict(round='round',model='model',mode='mode',status='ok',candidate_frozen=True,
        source=source['original_source'],api_calls=0,termination=None,terminal_budget_error=None,phases=[])
    for index,phase in enumerate(('calibration','prompt','step','step'),1):
        worker=dict(status='ok')
        write(case/'phases'/(str(index).zfill(3)+'-'+phase)/'worker-result.json',worker)
        result['phases'].append(dict(phase=phase,worker=worker,process=dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[],signals_sent=[])))
    write(case/'result.json',result);write(case/'completion.json',dict(status='ok',round='round',required_artifacts_persisted=True,registry_persisted=True))
    write(case/'actual-resources.jsonl',dict(processes=[dict(pid=999991,starttime=123)],outer_process=None))
    row=dict(run_id=case.name,round='round',model='model',mode='mode',git_rev='fixture',metrics=dict(status='ok'),artifacts_dir=str(case),ts='fixture')
    (case.parent/'registry.jsonl').write_text(json.dumps(row)+'\n')
    events=case/'phases/003-step/evaluation-stage-events.jsonl';events.parent.mkdir(parents=True,exist_ok=True)
    clock=iter(range(10000));recorder=StageRecorder(events,clock=lambda:next(clock))
    with recorder.span('cost'):
        for name in cache.NAMES:
            with recorder.span('component_cost_'+name):
                with recorder.span('cost_estimate'):
                    for stage in ('record_lower','placement','plan_express','conditional_plan_replay'):
                        with recorder.span(stage):pass
    return case,source,partition


class PlanCacheChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.case,self.source,self.partition=fixture(self.temp.name)

    def prepare(self):
        with patch('pt_agent.experiments.r10_paper_metrics._process',return_value=None):
            return cache.prepare_outer(self.case,self.source)

    def test_complete_typed_selected25_and_real_event_schema_admitted(self):
        proof=self.prepare()
        self.assertTrue(proof['registry_checked_by_outer']);self.assertTrue(proof['all_process_births_closed'])
        self.assertEqual(list(proof['facts']['components']),list(cache.NAMES))
        self.assertNotIn('plan',proof['facts']['components']['layer0'])

    def test_worker_uses_canonical_proof_without_registry_or_process_reads(self):
        proof=self.prepare();native=Path(self.temp.name)/'native';native.mkdir();write(native/'source-record.json',self.source);cache.persist(native,proof)
        args=SimpleNamespace(source_case=self.case,codegen=native/'codegen');recorder=StageRecorder(native/'events.jsonl')
        opened=Path.open
        def guarded(path,*args,**kwargs):
            if path.name=='registry.jsonl':raise AssertionError('worker touched registry')
            return opened(path,*args,**kwargs)
        with patch.object(Path,'open',guarded),patch('pt_agent.experiments.r10_paper_metrics._process',side_effect=AssertionError('worker process probe')):
            got=cache.worker_plan(args,self.source,self.partition.layers[0],recorder)
        self.assertTrue(got['provenance']['cache_hit']);self.assertFalse(got['provenance']['fresh_planner_executed'])
        self.assertFalse(got['verification']['fresh_verify_executed'])
        self.assertTrue((native/'qwen_component_plan_cache.py').is_file())
        self.assertEqual(got['plan']['calls'],35)

    def test_old_source_returns_fresh_route_without_artifact_reads(self):
        with patch.object(cache,'_artifacts',side_effect=AssertionError('old route cache read')):
            self.assertIsNone(cache.prepare_outer(self.case,dict(entry_module='old')))
            self.assertIsNone(cache.worker_plan(None,dict(entry_module='old'),None,None))

    def test_partial25_and_selected_cost_mutation_rejected(self):
        path=self.case/'session/component-cost-1/partial.json';saved=cache._read(path)
        write(path,dict(saved,completed=24))
        with self.assertRaises(ValueError):self.prepare()
        write(path,saved)
        path=self.case/'session/AGENT.json';saved=cache._read(path);saved['best']['boots_pred']+=1;write(path,saved)
        with self.assertRaises(ValueError):self.prepare()

    def test_unclosed_source_or_missing_c2_rejected(self):
        with patch('pt_agent.experiments.r10_paper_metrics._process',return_value=dict(starttime=123,state='S')):
            with self.assertRaises(ValueError):cache.prepare_outer(self.case,self.source)
        (self.case/'session/component-c2-access.json').unlink()
        with self.assertRaises(FileNotFoundError):self.prepare()

    def test_duplicate_registry_or_wrong_identity_rejected(self):
        registry=self.case.parent/'registry.jsonl';line=registry.read_text();registry.write_text(line+line)
        with self.assertRaises(ValueError):self.prepare()
        row=json.loads(line);row['artifacts_dir']=str(self.case.parent);registry.write_text(json.dumps(row)+'\n')
        with self.assertRaises(ValueError):self.prepare()

    def test_missing_or_failed_actual_verify_span_rejected(self):
        path=self.case/'phases/003-step/evaluation-stage-events.jsonl'
        events=[json.loads(line)for line in path.read_text().splitlines()]
        target=next(e for e in events if e['stage_id']=='conditional_plan_replay'and e['type']=='end')
        target['status']='error';path.write_text('\n'.join(json.dumps(e)for e in events)+'\n')
        with self.assertRaises(ValueError):self.prepare()

    def test_plan_schema_and_boot_coverage_reject_malformed_keys(self):
        path=self.case/'session/component-cost-1/layer0/refresh-plan.json';original=cache._read(path)
        for change in (dict(dead=[[34,36]]),dict(substitutions=[[1,0,3],[1,0,3]]),dict(boots=1),dict(calls=True),dict(outputs=[[34,1]])):
            with self.subTest(change=change),self.assertRaises(ValueError):cache.validate_plan(dict(original,**change),34)

    def test_wrong_typed_source_binding_and_path_escape_rejected(self):
        path=self.case/'session/component-cost-1/layer0/source-binding.json';binding=cache._read(path)
        binding['fixture_foreign']=True;write(path,binding)
        with self.assertRaises(ValueError):self.prepare()
        with self.assertRaises(ValueError):cache._inside(self.case.parent/'foreign',self.case/'session')

    def test_worker_revalidates_current_stat_and_full_artifacts(self):
        proof=self.prepare();native=Path(self.temp.name)/'native';native.mkdir();write(native/'source-record.json',self.source);cache.persist(native,proof)
        path=self.case/'session/component-cost-1/layer0/refresh-plan.json';plan=cache._read(path);plan['timing']={'lower_s':1.};write(path,plan)
        with self.assertRaises(ValueError):cache.worker_plan(SimpleNamespace(source_case=self.case,codegen=native/'codegen'),self.source,self.partition.layers[0],StageRecorder(native/'events.jsonl'))

    def test_fresh_count_requires_entire_stats_and_interface_match(self):
        original=cache._read(self.case/'session/component-cost-1/layer0/complete-count.json')
        cache.compare_count(original,copy.deepcopy(original))
        for key in ('stats','interface'):
            changed=copy.deepcopy(original);changed[key]['foreign']=1
            with self.subTest(key=key),self.assertRaises(ValueError):cache.compare_count(original,changed)

    def test_source_proof_preserves_existing_and_never_claims_fresh_clock(self):
        proof=self.prepare();native=Path(self.temp.name)/'native';native.mkdir();write(native/'source-record.json',self.source);cache.persist(native,proof)
        with self.assertRaises(ValueError):cache.persist(native,proof)
        self.assertIn('Historical',proof['facts']['clock_scope'])
        self.assertEqual(cache._read(native/cache.PROOF)['facts']['source'],self.source)

    def test_actual_two_placement_dispatches_cache_then_fresh_count_without_planner(self):
        from pt_agent.experiments import qwen_layer_native_probe as probe
        from pt_agent.experiments import qwen_remaining_prefill_compile as remaining
        from pt_agent.experiments import qwen_layered_structure as layered
        root=Path(self.temp.name)
        for module,name,segment in ((probe,'layer0',self.partition.layers[0]),
                                    (remaining,'layer23',self.partition.layers[23]),
                                    (remaining,'head',self.partition.head)):
            with self.subTest(component=name):
                directory=root/name;directory.mkdir()
                args=SimpleNamespace(codegen=directory,component=name)
                callback=layered.SegmentLowering(segment,{})
                count=cache._read(self.case/'session/component-cost-1'/name/'complete-count.json')
                plan=cache._read(self.case/'session/component-cost-1'/name/'refresh-plan.json')
                reused=dict(plan=plan,verification=dict(boots=0),count=count,
                    provenance=dict(cache_hit=True,fresh_planner_executed=False,fresh_verify_executed=False))
                class Count:
                    def __init__(self,*args,**kwargs):pass
                    def stats(self):return copy.deepcopy(count['stats'])
                patches=[patch.object(cache,'worker_plan',return_value=reused),
                    patch.object(probe,'planner_namespace',side_effect=AssertionError('fresh planner on cache hit')),
                    patch.object(remaining.original,'_planner_namespace',side_effect=AssertionError('fresh head planner on cache hit'))]
                from contextlib import ExitStack
                with ExitStack() as stack:
                    for item in patches:stack.enter_context(item)
                    if module is probe:
                        stack.enter_context(patch.object(probe,'_load',return_value=(self.partition,callback)))
                        counted=stack.enter_context(patch.object(probe,'_count',return_value=count))
                    else:
                        stack.enter_context(patch.object(remaining,'_load',return_value=callback))
                        stack.enter_context(patch('pt_agent.backends.streaming_count.StreamingCountBackend',Count))
                        counted=stack.enter_context(patch.object(remaining,'export_component',return_value=SimpleNamespace(metadata=count['interface'])))
                    result=dict(source=self.source)
                    module.placement(args,result,StageRecorder(directory/'events.jsonl'))
                    counted.assert_called_once()
                saved=cache._read(directory/'planner-return.json')
                self.assertTrue(saved['plan_cache']['cache_hit'])
                self.assertFalse(saved['plan_cache']['fresh_verify_executed'])
                self.assertTrue(result['placement_completed'])

    def test_outer_probe_source_record_seam_writes_proof_without_source_mutation(self):
        from pt_agent.experiments import qwen_layer_native_probe as probe
        output=Path(self.temp.name)/'native';namespace=dict(frozen_source=lambda case:self.source,write_json=write)
        def execute(args):
            admitted=namespace['frozen_source'](args.source_case)
            args.out.mkdir();namespace['write_json'](args.out/'source-record.json',admitted)
            return admitted
        namespace['execute']=execute
        old_timer=__import__('signal').getitimer(__import__('signal').ITIMER_REAL)
        try:
            with patch.object(probe,'_namespace',return_value=namespace), \
                    patch('pt_agent.experiments.r10_paper_metrics._process',return_value=None):
                returned=probe.execute(SimpleNamespace(source_case=self.case,out=output,seconds=60))
        finally:
            __import__('signal').setitimer(__import__('signal').ITIMER_REAL,*old_timer)
        self.assertEqual(returned,self.source)
        self.assertEqual(cache._read(output/'source-record.json'),self.source)
        self.assertTrue(cache._read(output/cache.PROOF)['registry_checked_by_outer'])

    def test_json_saved_level_hist_keys_and_fresh_integer_keys_match(self):
        cache.compare_count(dict(stats=dict(counts={}),interface=dict(refresh=dict(level_hist={'5':2}))),
            dict(stats=dict(counts={}),interface=dict(refresh=dict(level_hist={5:2}))))

    def test_unclean_saved_phase_and_real_second_event_stream_handled(self):
        path=self.case/'result.json';record=cache._read(path)
        record['phases'][0]['process']['signals_sent']=[dict(signal=15)]
        write(path,record)
        with self.assertRaises(ValueError):self.prepare()
        record['phases'][0]['process']['signals_sent']=[];write(path,record)
        other=self.case/'phases/004-step/evaluation-stage-events.jsonl'
        with StageRecorder(other).span('heldout_validation'):pass
        self.assertTrue(self.prepare()['all_process_births_closed'])

    def test_outer_remaining_source_record_seam_keeps_original_source(self):
        from pt_agent.experiments import qwen_remaining_prefill_compile as remaining
        # Clone this small stand-in through the production outer namespace seam.
        def execute(args):
            admitted=frozen_source(args.source_case)
            args.out.mkdir();write_json(args.out/'source-record.json',admitted)
            return admitted
        output=Path(self.temp.name)/'remaining-native'
        args=SimpleNamespace(source_case=self.case,out=output,seconds=60,component='layer23')
        old_timer=__import__('signal').getitimer(__import__('signal').ITIMER_REAL)
        try:
            with patch.object(remaining.original,'execute',execute), \
                    patch.object(remaining,'_component_source',return_value=self.source), \
                    patch('pt_agent.experiments.r10_paper_metrics._process',return_value=None):
                returned=remaining.component_execute(args)
        finally:
            __import__('signal').setitimer(__import__('signal').ITIMER_REAL,*old_timer)
        self.assertEqual(returned,self.source)
        self.assertEqual(cache._read(output/'source-record.json'),self.source)
        self.assertTrue(cache._read(output/cache.PROOF)['registry_checked_by_outer'])

    def test_real_tiny_original_planner_and_complete_count_saved_schema_match(self):
        import numpy as np
        from pt_agent.experiments import qwen_layered_structure as layered
        from pt_agent.experiments import qwen_layer_native_probe as probe
        from pt_agent.experiments.qwen_component_cost import already_scoped_lowering
        segment=self.partition.layers[0]
        weights={name:np.arange(shape[0],dtype=np.float64)*2.**-20 for name,shape in segment.weight_shapes}
        planner=probe.planner_namespace()
        from pt_agent.kernels.compact_linear import compact_transformer_projections
        with probe.original._projection_backend_scope(),compact_transformer_projections():
            plan,dag,placed=planner['plan_program'](segment.program,weights,profile='qwen48',frontend_normalize=True)
            verification=planner['verify_program'](segment.program,weights,plan,profile='qwen48')
        cache.validate_plan(plan,34)
        callback=already_scoped_lowering(layered.SegmentLowering(segment,weights))
        bound=layered.bind_plan(segment,plan,planned_source=layered.source_binding(segment))
        counted=probe._count(callback,bound)
        self.assertEqual(counted['stats']['counts']['boot'],verification['boots']+32)
        self.assertEqual(counted['interface']['symbolic_output_levels'],[5]*32)
        saved=cache._same(counted)
        cache.compare_count(saved,counted)

    def test_worker_outer_registry_identity_and_canonical_source_record_required(self):
        proof=self.prepare();native=Path(self.temp.name)/'proof-native';native.mkdir()
        write(native/'source-record.json',self.source);cache.persist(native,proof)
        args=SimpleNamespace(source_case=self.case,codegen=native/'codegen')
        got=cache.worker_plan(args,self.source,self.partition.layers[0],StageRecorder(native/'events.jsonl'))
        self.assertTrue(got['provenance']['cache_hit'])
        proof['registry']['run_id']='foreign';write(native/cache.PROOF,proof)
        with self.assertRaises(ValueError):cache.worker_plan(args,self.source,self.partition.layers[0],StageRecorder(native/'events.jsonl'))
        write(native/'source-record.json',dict(self.source,selected_iteration=12))
        with self.assertRaises(ValueError):cache.worker_plan(args,self.source,self.partition.layers[0],StageRecorder(native/'events.jsonl'))


if __name__=='__main__':unittest.main()
