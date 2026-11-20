"""BERT publication identity/clock separation, without experiments."""
import copy
import unittest
from pt_agent.experiments import bert_budget_paper_projection as entry


def fixture():
    case='/fixture/'+entry.NATIVE2
    row=dict(run_id=entry.NATIVE2,source=case+'/result.json',model='bert_tiny',system='dacapo_search_ours',
             param_group='ours-N16-boot5',budget_s=7200,status='timeout',compile_s=7200.4)
    native=dict(started_ns=10**9,ended_ns=7201400000000,timed_out=True,exit_code=-15,remaining_pids=[],
                limits=dict(wall_seconds=7200,memory_gib=24),signals_sent=[dict(pid=12,signal=15)])
    result=dict(status='error',phases=dict(compile=dict(native_process=native)))
    done=dict(status='error',registry_persisted=True,required_artifacts_persisted=True,
              complete_case_elapsed_seconds=7224.)
    proof=dict(paper_record_original=copy.deepcopy(row),registry_fields=8,registry_run_id=entry.NATIVE2,
               source_result=case+'/result.json',source_completion=case+'/completion.json',producer_closure=dict(closed=True))
    comparison=dict(schema='bert-two-hour-fresh-vs-independent-native-comparison-v1',
        dacapo_current_rule63_search=dict(state='terminal',classification='timeout',native_pipeline_proof=proof),
        fresh_arranger=dict(state='terminal',usable_within_two_hours=False,case='/fixture/'+entry.FRESH2))
    selector=dict(cells=[dict(model='bert_tiny',system='dacapo_search_ours',selected_run_id=None),
                         dict(model='bert_tiny',system='arranger',scientific_run_id='retained',boots_compiled=9180,generation_e2e_s=None),
                         dict(model='resnet20',system='orion',compile_s=None,cell_state='terminal')])
    return selector,comparison,row,result,done


class BertBudgetProjectionTests(unittest.TestCase):
    def test_only_measured_native_cell_changes_and_warm_program_remains_separate(self):
        args=fixture();original=copy.deepcopy(args)
        def selected(cell,row,result,done):cell.update(selected_run_id=row['run_id'],clock_kind='cold',compilation_status=row['status'])
        value=entry.project(*args,selected=selected)
        self.assertEqual(args,original)
        self.assertEqual(value['cells'][1:],original[0]['cells'][1:])
        cell=value['cells'][0];self.assertEqual(cell['clock_kind'],'actual_compile_only')
        self.assertIsNone(cell['generation_e2e_s']);self.assertIsNone(cell['boots_compiled'])
        self.assertEqual(cell['actual_compile_only_case_s'],7224.)
        self.assertFalse(value['BERT_budget_observations']['three_hour_outcomes_assumed'])

    def test_unclosed_or_changed_original_record_is_rejected(self):
        for change in('birth','row','clock'):
            args=list(fixture())
            if change=='birth':args[1]['dacapo_current_rule63_search']['native_pipeline_proof']['producer_closure']['closed']=False
            elif change=='row':args[2]['status']='compiler_crash'
            else:args[3]['phases']['compile']['native_process']['ended_ns']-=10**9
            with self.assertRaises(ValueError):entry.project(*args,selected=lambda *unused:None)

    def test_later_independent_native_cell_is_never_replaced(self):
        args=list(fixture());args[0]['cells'][0]['selected_run_id']='later-independent-three-hour-result'
        with self.assertRaisesRegex(ValueError,'later BERT'):entry.project(*args,selected=lambda *unused:None)


if __name__=='__main__':unittest.main()
