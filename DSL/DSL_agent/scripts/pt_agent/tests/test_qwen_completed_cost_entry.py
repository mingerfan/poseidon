"""New optional completed-cost entry/profiles/authority fixtures; no model/API."""
import copy
import inspect
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_retained_component_replay as entry
from pt_agent.experiments import qwen_component_plan_cache as cache
from pt_agent.experiments import qwen_agent_cost_lifetime as lifetime
from pt_agent.experiments.r10_paper_metrics import StageRecorder


def evaluator(functions):
    # Original bridge guard owns the original private agent_tools evaluator.
    return inspect.getclosurevars(functions['evaluate']).nonlocals['original_evaluate']


def adapted_tools(functions):
    measured=evaluator(functions)
    return inspect.getclosurevars(measured).nonlocals['evaluate']


def args_for(root):
    return SimpleNamespace(completed_cost_from=Path(root)/'old',logical_constant_inventory_gib=32,
        completed_cost_admission=Path(root)/'completed-cost-cache-admission.json',
        calibration=Path(root)/'calibration.json',partial_from=Path(root)/'paid',
        session=Path(root)/'session',worker_phase='step')


class CompletedEntryChecks(unittest.TestCase):
    def test_optional_cli_defaults_pair_and_exact32_policy(self):
        parser=entry.parser()
        for name in ('completed_cost_from','logical_constant_inventory_gib','completed_cost_admission'):
            self.assertIsNone(parser.get_default(name))
        self.assertFalse(entry._completed_enabled(SimpleNamespace()))
        for case,amount in ((None,32),(Path('/tmp/old'),None),(Path('/tmp/old'),4),(Path('/tmp/old'),True)):
            with self.subTest(case=case,amount=amount),self.assertRaises(ValueError):
                entry._completed_enabled(SimpleNamespace(completed_cost_from=case,logical_constant_inventory_gib=amount))
        self.assertTrue(entry._completed_enabled(args_for('/tmp/fresh')))

    def test_private_profile_changes_only_logical_inventory_original_unchanged(self):
        from pt_agent.cost import PROFILES
        before=copy.deepcopy(PROFILES)
        original=entry.guarded.bridge._agent_functions()
        private=entry._inventory_functions(original)
        inner=evaluator(private)
        self.assertEqual(inner.__globals__['PROFILES']['qwen48']['limits']['const_bytes'],32*entry.GIB)
        expected=copy.deepcopy(before);expected['qwen48']['limits']['const_bytes']=32*entry.GIB
        self.assertEqual(inner.__globals__['PROFILES'],expected)
        self.assertEqual(PROFILES,before)
        self.assertEqual(evaluator(original).__globals__['PROFILES']['qwen48']['limits']['const_bytes'],4*entry.GIB)
        self.assertIs(private['agent_step'].__globals__,private)

    def test_original_evaluate_exact_cost_then_heldout_with_private_limit(self):
        original=entry.guarded.bridge._agent_functions();private=entry._inventory_functions(original)
        for functions,accepted in ((original,False),(private,True)):
            call=evaluator(functions);sequence=[];namespace=call.__globals__
            namespace.update(parse=lambda source:SimpleNamespace(ops=[]),
                check=lambda *a,**k:SimpleNamespace(ok=True),
                _exact_check=lambda *a,**k:(sequence.append('fresh_exact') or dict(ok=True)),
                cost_feedback=lambda *a,**k:(sequence.append('completed_cost') or (dict(const_bytes=5*entry.GIB,rot_keys=0,boot_nodes=[]),{})),
                validate=lambda *a,**k:(sequence.append('fresh_heldout') or dict(feasible=True,diverged=False)))
            feedback=call('synthetic source',SimpleNamespace(profile='qwen48',weight_shapes={}))
            self.assertEqual(feedback['feasible'],accepted)
            self.assertEqual(sequence,['fresh_exact','completed_cost']+(['fresh_heldout'] if accepted else []))
            if not accepted:self.assertEqual(feedback['errors'][0]['limit'],'const_bytes')

    def test_actual_adapted_agent_dispatch_uses_private_profile_and_cached_estimator(self):
        with tempfile.TemporaryDirectory() as temp:
            args=args_for(temp)
            functions=entry.agent_functions(temp,lifetime.Diagnostics(),completed_args=args,completed_source={})
            tools=adapted_tools(functions)
            self.assertEqual(tools.__globals__['PROFILES']['qwen48']['limits']['const_bytes'],32*entry.GIB)
            self.assertIs(functions['agent_step'].__globals__,functions)
            self.assertIs(tools.__globals__['estimate'],tools.__globals__['estimate_normalized'])
            self.assertIs(tools.__globals__['estimate_fast'],tools.__globals__['estimate_normalized_fast'])

    def test_default_agent_dispatch_stays_original4_and_original_component_estimator(self):
        with tempfile.TemporaryDirectory() as temp:
            functions=entry.agent_functions(temp,lifetime.Diagnostics())
            tools=adapted_tools(functions)
            self.assertEqual(tools.__globals__['PROFILES']['qwen48']['limits']['const_bytes'],4*entry.GIB)
            with patch.object(entry.component,'estimate',return_value={'original':True}) as measured:
                answer=tools.__globals__['estimate'](object(),object(),profile='qwen48')
            self.assertEqual(answer,{'original':True});measured.assert_called_once()
            self.assertEqual(measured.call_args.kwargs['directory'],Path(temp)/'component-cost-1')

    def test_cached_estimator_original_cost_payload_no_new_Count_and_source_bound(self):
        from pt_agent.dsl.parser import parse
        text=('@fhe.program(slots=1024)\n'
            'def tiny(x: Cipher[1]):\n    out=add(x,x)\n    return out\n')
        program=parse(text)
        with tempfile.TemporaryDirectory() as temp:
            args=args_for(temp);provenance=dict(authority_case=str(args.completed_cost_from),fresh_Count_executed=False)
            cost=dict(const_bytes=5*entry.GIB,refresh_plan=dict(components=[]))
            diagnostics=lifetime.Diagnostics(Path(temp)/'events.jsonl')
            functions=entry.agent_functions(temp,diagnostics,completed_args=args,completed_source={'authority':'expected'})
            estimate=adapted_tools(functions).__globals__['estimate']
            with patch.object(entry,'_retained_source_text',return_value=text), \
                    patch.object(entry,'_completed_load',return_value=dict(cost=cost,provenance=provenance)) as loaded, \
                    patch.object(entry.component,'estimate',side_effect=AssertionError('fresh Count forbidden')):
                answer=estimate(program,{},profile='qwen48')
            self.assertEqual(answer,cost);self.assertIsNot(answer,cost);loaded.assert_called_once_with(args,{'authority':'expected'})
            self.assertEqual(json.loads((Path(temp)/'completed-cost-reuse.json').read_text()),provenance)
            self.assertFalse((Path(temp)/'component-cost-1').exists())
            events=[json.loads(line) for line in (Path(temp)/'events.jsonl').read_text().splitlines()]
            self.assertTrue(all(e['cache_hit'] for e in events))
            self.assertEqual({e['stage_id']for e in events},{'completed_component_cost_cache_hit'})
            with patch.object(entry,'_retained_source_text',return_value=text.replace('add(x,x)','mul(x,x)')), \
                    patch.object(entry,'_completed_load',side_effect=AssertionError('foreign program cache access')):
                with self.assertRaises(ValueError):estimate(program,{},profile='qwen48')

    def test_phase_argv_new_flags_and_canonical_proof_default_argv_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            args=args_for(temp);args.resume_from=Path(temp)/'resume';args.grouped_from=Path(temp)/'grouped'
            args.rss_gib=16;args.minimum_available_gib=1.25;args.seconds=9000
            namespace=dict(PHASE_SECONDS=dict(step=3600),phase_command=lambda *a:['original-command'])
            fake=ModuleType('pt_agent.experiments.qwen_completed_cost_cache');fake.PROOF='completed-cost-cache-admission.json'
            with patch.object(entry.guarded.bridge,'_namespace',return_value=namespace),patch.dict(sys.modules,{fake.__name__:fake}):
                command=entry.phase_command(args,'step',Path(temp)/'work')
                self.assertEqual(command[command.index('--logical-constant-inventory-gib')+1],'32')
                self.assertEqual(command[command.index('--completed-cost-admission')+1],str(Path(temp)/fake.PROOF))
                args.completed_cost_from=None;args.logical_constant_inventory_gib=None;args.completed_cost_admission=None
                default=entry.phase_command(args,'step',Path(temp)/'work')
            self.assertNotIn('--completed-cost-from',default)
            self.assertEqual(command[:command.index('--completed-cost-from')],default)

    def test_source_gate_full_original_precedes_new_outer_proof_and_inner_readonly(self):
        import pt_agent.experiments as package
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=args_for(temp);args.worker_phase=None;args.out=None
            args.source_admission=root/'component-source-admission.json'
            fake=ModuleType('pt_agent.experiments.qwen_completed_cost_cache');fake.PROOF='completed-cost-cache-admission.json';sequence=[]
            fake.prepare_outer=lambda case,source:(sequence.append(('outer',case,copy.deepcopy(source))) or {'proof':True})
            def persist(directory,proof):
                path=Path(directory)/fake.PROOF;path.write_text(json.dumps(proof));return path
            fake.persist=persist
            fake.load_worker=lambda *a:(sequence.append(('inner',a[0],copy.deepcopy(a[2]))) or dict(cost={},provenance=dict(authority_case=str(args.completed_cost_from))))
            with patch.object(package,'qwen_completed_cost_cache',fake,create=True),patch.dict(sys.modules,{fake.__name__:fake}), \
                    patch.object(entry,'check_write_paths'),patch.object(entry.guarded,'source_gate',return_value={'original':1}), \
                    patch.object(entry,'_closed_grouped',return_value={'closed':True}), \
                    patch.object(entry,'_retained_source_text',return_value='retained source'):
                outer=entry.source_gate(args)
                self.assertEqual(sequence[0][0],'outer')
                self.assertNotIn('completed_cost_authority',sequence[0][2])
                self.assertEqual(outer['logical_constant_inventory_gib'],32)
                args.worker_phase='step';fake.prepare_outer=lambda *a:(_ for _ in ()).throw(AssertionError('inner outer gate'))
                inner=entry.source_gate(args)
                self.assertEqual(inner,outer)
                self.assertEqual(entry._completed_base_source(outer),sequence[0][2])

    def test_native_cache_new_C2_selection_keeps_old_cost_and_event_authority(self):
        import pt_agent.experiments as package
        from pt_agent.tests.test_qwen_component_plan_cache import fixture,write
        from pt_agent.planners.transformer_session import load_state,save_state,json_safe
        with tempfile.TemporaryDirectory() as temp:
            case,source,partition=fixture(temp);old=Path(temp)/'cost-authority';old.mkdir();(old/'session').mkdir()
            shutil.copytree(case/'session/component-cost-1',old/'session/component-cost-1')
            shutil.copytree(case/'phases',old/'phases')
            cost=cache._read(old/'session/component-cost-1/result.json')
            for row in cost['refresh_plan']['components']:
                row['plan']=str(old/'session/component-cost-1'/row['name']/'refresh-plan.json')
            write(old/'session/component-cost-1/result.json',cost)
            payload={k:v for k,v in cost.items() if k!='status'}
            state=load_state(case/'session/SESSION.json');key=next(iter(state['records']))
            record=dict(payload,iteration=13,source='llm');state['records'][key]=(record,1);state['best']=record
            save_state(case/'session/SESSION.json',state)
            agent=cache._read(case/'session/AGENT.json');agent['best']=json_safe(record);write(case/'session/AGENT.json',agent)
            provenance=dict(authority_case=str(old),authority_cost_directory=str(old/'session/component-cost-1'),
                fresh_Count_executed=False,heldout_or_C2_acceptance_inherited=False)
            expected=copy.deepcopy(source['original_source'])
            source['original_source'].update(completed_cost_authority=provenance,
                completed_cost_admission=str(case/'completed-cost-cache-admission.json'),
                logical_constant_inventory_gib=32,logical_constant_inventory_scope=entry.INVENTORY_SCOPE)
            saved=cache._read(case/'result.json');saved['source']=source['original_source'];write(case/'result.json',saved)
            write(case/'completed-cost-cache-admission.json',{'proof':'old complete cost'})
            fake=ModuleType('pt_agent.experiments.qwen_completed_cost_cache');fake.PROOF='completed-cost-cache-admission.json'
            calls=[]
            def load(oldcase,path,base,text):
                self.assertEqual(oldcase,old);self.assertEqual(base,expected)
                self.assertEqual(path,case/fake.PROOF)
                return dict(cost=payload,provenance=provenance)
            fake.load_worker=load;fake.prepare_outer=lambda *a:calls.append(a)
            with patch.object(package,'qwen_completed_cost_cache',fake,create=True),patch.dict(sys.modules,{fake.__name__:fake}), \
                    patch('pt_agent.experiments.r10_paper_metrics._process',return_value=None):
                proof=cache.prepare_outer(case,source)
            self.assertEqual(calls,[(old,expected)])
            self.assertEqual(proof['facts']['source_case'],str(case))
            self.assertEqual(proof['facts']['cost_directory'],str(old/'session/component-cost-1'))
            self.assertTrue(proof['facts']['events']['path'].startswith(str(old)))
            self.assertEqual(proof['facts']['completed_cost_authority'],provenance)

    def test_foreign_inner_completed_proof_path_rejected_before_helper_load(self):
        import pt_agent.experiments as package
        fake=ModuleType('pt_agent.experiments.qwen_completed_cost_cache');fake.PROOF='completed-cost-cache-admission.json'
        fake.load_worker=lambda *a:(_ for _ in ()).throw(AssertionError('foreign proof loaded'))
        with tempfile.TemporaryDirectory() as temp:
            args=args_for(temp);args.completed_cost_admission=Path(temp)/'foreign.json'
            with patch.object(package,'qwen_completed_cost_cache',fake,create=True),patch.dict(sys.modules,{fake.__name__:fake}):
                with self.assertRaises(ValueError):entry._completed_load(args,{})


if __name__=='__main__':unittest.main()
