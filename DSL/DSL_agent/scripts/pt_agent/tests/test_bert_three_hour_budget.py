"""Three-hour budget and semantic-regression checks; scientific execution zero."""
import ast
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from pt_agent.experiments import bert_fresh_cold_2h as old_agent
from pt_agent.experiments import bert_fresh_cold_3h as agent
from pt_agent.experiments import bert_rule63_shared_optimized_compile as old_native
from pt_agent.experiments import bert_rule63_shared_optimized_compile_3h as native


class Normalize(ast.NodeTransformer):
    """Compare trusted numerical/control code after explicit budget renaming."""
    def __init__(self, flavour):self.flavour=flavour

    def name(self,value):return value.replace('three_hour','two_hour')

    def visit_Name(self,node):
        node.id=self.name(node.id);return node

    def visit_keyword(self,node):
        if node.arg:node.arg=self.name(node.arg)
        return self.generic_visit(node)

    def visit_List(self,node):
        if self.flavour=='agent':
            values=node.elts;kept=[];index=0
            while index<len(values):
                value=values[index]
                if (isinstance(value,ast.Constant)and value.value=='--step-seconds'
                        and index+1<len(values)and isinstance(values[index+1],ast.Constant)
                        and values[index+1].value=='10800'):
                    index+=2;continue
                kept.append(value);index+=1
            node.elts=kept
        return self.generic_visit(node)

    def visit_FunctionDef(self,node):
        node.name=self.name(node.name)
        return self.generic_visit(node)

    def visit_Constant(self,node):
        value=node.value
        if type(value)is int:
            node.value=({10800:7200,9000:5400}if self.flavour=='agent'else{10800:7200,13200:9600}).get(value,value)
        elif isinstance(value,str):
            if self.flavour=='agent':
                value=value.replace(agent.MODE,old_agent.MODE).replace(agent.MODULE,old_agent.MODULE)
                value=value.replace('three-hour','two-hour').replace('three_hour','two_hour')
                value=value.replace('10800','7200').replace('9000','5400').replace('-3H','-2H').replace('-3h','-2h')
            else:
                value=value.replace(native.MODE,old_native.MODE).replace(native.MODULE,old_native.MODULE)
                value=value.replace('10800','7200').replace('13200','9600')
            node.value=value
        return node


def functions(module):
    return {n.name:n for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}


class ThreeHourTests(unittest.TestCase):
    def observation(self,seconds,*,accepted=True,observed=None,registry=True):
        start=100;result=dict(compile_accepted=accepted,native_completed_ns=start+int(seconds*1e9))
        completion=dict(required_artifacts_persisted=True,registry_persisted=registry,
            ended_ns=start+int(seconds*1e9),complete_case_elapsed_seconds=seconds)
        return agent.three_hour_observation(result,completion,start,start+int((seconds if observed is None else observed)*1e9))

    def test_budget_changes_preserve_original_tail_and_model_policy(self):
        self.assertEqual((agent.TOTAL_SECONDS,agent.SEARCH_CEILING,agent.TAIL_RESERVE),(10800,9000,1800))
        self.assertEqual((old_agent.TOTAL_SECONDS,old_agent.SEARCH_CEILING),(7200,5400))
        self.assertEqual(agent.MODEL,old_agent.MODEL)
        self.assertNotEqual(agent.ROUND,old_agent.ROUND)

    def test_two_hour_and_three_hour_completion_are_separate_actual_observations(self):
        early=self.observation(7199);late=self.observation(8000)
        self.assertTrue(early['persisted_pipeline_within_two_hours'])
        self.assertTrue(early['persisted_pipeline_within_three_hours'])
        self.assertFalse(late['persisted_pipeline_within_two_hours'])
        self.assertTrue(late['persisted_pipeline_within_three_hours'])
        self.assertEqual(late['budget_seconds'],10800)

    def test_late_observation_and_missing_persistence_reject_three_hour_acceptance(self):
        for value in [self.observation(10799,observed=10801),self.observation(8000,registry=False),
                      self.observation(8000,accepted=False)]:
            self.assertFalse(value['persisted_pipeline_within_three_hours'])
            self.assertFalse(value['persisted_pipeline_within_two_hours'])

    def test_original_agent_scientific_control_and_workers_match_semantically(self):
        old,new=functions(old_agent),functions(agent)
        for name,node in old.items():
            if name in('two_hour_observation','paper_record'):continue
            copied=Normalize('agent').visit(copy.deepcopy(new[name]))
            self.assertEqual(ast.dump(node,include_attributes=False),ast.dump(copied,include_attributes=False),name)

    def test_original_native_algorithm_source_and_full_configuration_match_semantically(self):
        old,new=functions(old_native),functions(native)
        for name,node in old.items():
            copied=Normalize('native').visit(copy.deepcopy(new[name]))
            self.assertEqual(ast.dump(node,include_attributes=False),ast.dump(copied,include_attributes=False),name)

    def test_native_dispatch_receives_full10800_and_separate_outer_cleanup_time(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);(root/'source').mkdir()
            (root/'source/shared-generation-source.json').write_text(json.dumps(dict(input_kind='raw-symlink')))
            args=SimpleNamespace(out=root,profile=root/'profile.json',metadata_template=root/'template.json',
                                 case_gib=32,diagnostic_elide_final_ir=True)
            argv,outer=native.phase_command(args,'compile',13000)
            self.assertIn(native.MODULE,argv)
            self.assertEqual(argv[argv.index('--native-seconds')+1],'10800')
            self.assertEqual(argv[argv.index('--seconds')+1],'10860')
            self.assertEqual(outer,10920)

    def test_native_parser_preserves_finite_three_hour_budget(self):
        with TemporaryDirectory()as folder:
            argv=['--out',str(Path(folder)/'new'),'--paper-runs',str(Path(folder)/'runs.jsonl'),
                  '--host-pid-namespace','fixture']
            args=native.arguments(argv)
            self.assertEqual(args.native_seconds,10800);self.assertEqual(args.seconds,13200)
            with self.assertRaises(ValueError):native.arguments(argv+['--native-seconds','10801'])

    def test_original_recorded_worker_receives_declared_three_hour_nominal_allowance(self):
        with TemporaryDirectory()as folder:
            root=Path(folder)
            args=SimpleNamespace(metadata_template=root/'template.json',model=root/'model',inputs=root/'inputs',
                out=root/'new',ledger=root/'ledger.jsonl',allowance=root/'allowance.json',
                data_python=Path('/fixture/data-python'),intake_python=Path('/fixture/intake-python'))
            argv=agent.command(args,'step',root/'work',8900)
            self.assertIn('pt_agent.experiments.bert_sst2_recorded',argv)
            self.assertEqual(argv[argv.index('--step-seconds')+1],'10800')
            self.assertEqual(argv[argv.index('--seconds')+1],'8900')
            self.assertNotIn('pt_agent.experiments.bert_ordered_tuner_step',argv)

    def test_paper_row_distinguishes_inner_search_timeout_from_complete_three_hour_use(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);out=root/'new';out.mkdir()
            args=SimpleNamespace(out=out,profile=root/'profile.json',paper_runs=root/'runs.jsonl')
            observation=dict(exit_code=124,timed_out=False,signal=None,signals_sent=[],remaining_pids=[],
                             started_ns=100,ended_ns=100+9000*10**9)
            result=dict(status='error',error='fixture inner search timeout',compile_accepted=False,
                processes=[dict(phase='step',argv=['python','-m','pt_agent.tools.nixrun','--seconds','9000'],
                                observation=observation)],phases={},timing=dict(spans=[],attribution_ready=True),
                wall_policy=dict(search_stop='fixture only'),api_calls=0)
            record=dict(persisted_pipeline_within_three_hours=False,persisted_pipeline_within_two_hours=False,
                elapsed_seconds=9001.,completion_elapsed_seconds=9000.5,usable_definition='fixture only',status='error')
            agent.paper_record(args,result,record)
            row=json.loads(args.paper_runs.read_text())
            self.assertEqual(row['budget_s'],10800)
            self.assertEqual(row['status'],'timeout')
            self.assertEqual(row['timeout_scope'],'inner_phase_allocation')
            self.assertEqual(row['inner_phase_timeouts'][0]['allocated_seconds'],9000)
            self.assertFalse(row['complete_10800_seconds_consumed'])
            self.assertFalse(row['usable_within_two_hours'])
            self.assertFalse(row['usable_within_three_hours'])


if __name__=='__main__':unittest.main()
