"""Provider failures retain partial records without rerunning computations."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from pt_agent.experiments import e6r_llm_select_fix as entry


def fixture(root):
    session=entry.session_path(root,'e6r-a','bert','E6r-a')
    entry.write(session/'REPORT.json',dict(complete=False,failure_class='provider',error='LLMError: invalid_response',seconds=50))
    entry.write(session/'steps/005-tune/REPORT.json',dict(complete=False,failure_class='provider',error='LLMError: invalid_response',seconds=40))
    rounds=[dict(round=i,candidates=[dict(moves=['m1'],evaluated=True,accepted=True,
        latency_pred_us=10-i,full=dict(feasible=True))],finish_reason='stop') for i in (1,2)]
    folder=root/'calls/bert/t1/E6r-a'
    for n in (1,2,3):
        entry.write(folder/f'call-{n:03d}.request.json',dict(purpose=f'tune-select-iter4-round{n}',
            messages=[dict(content='system'),dict(content=json.dumps(dict(menu=[dict(id='m1')],previous_rounds=rounds[:n-1])))]))
        if n<3:entry.write(folder/f'call-{n:03d}.response.json',dict(seconds=n,usage=dict(prompt_tokens=5,completion_tokens=2)))
    entry.write(folder/'call-003.error.json',dict(error='invalid_response',diagnostics=dict(status=200,elapsed_seconds=3)))
    return session


class ReportingTests(unittest.TestCase):
    def test_completed_rounds_recovered_and_failed_round_recorded(self):
        with TemporaryDirectory() as folder:
            session=fixture(Path(folder));rows=entry.selector_rounds(session)
            self.assertEqual([r['iteration'] for r in rows],[4,4,4])
            self.assertEqual([r['history']['round'] for r in rows],[1,2,3])
            self.assertEqual([r['record_source'] for r in rows],['saved_next_request','saved_next_request','provider_error'])
            self.assertTrue(rows[0]['history']['candidates'][0]['accepted'])
            self.assertEqual(rows[2]['history']['error'],'invalid_response')
            self.assertFalse((session/'FROZEN.json').exists())

    def test_failure_times_tokens_and_partial_tune_stay_distinct(self):
        with TemporaryDirectory() as folder:
            session=fixture(Path(folder));row=entry.session_summary(session)
            self.assertEqual(row['acceptance_status'],'provider_failed')
            self.assertEqual(row['session_failure_class'],'provider')
            self.assertEqual((row['main_calls'],row['tuning_calls']),(0,3))
            self.assertEqual(row['timings']['tuning_llm_s'],6)
            self.assertEqual(row['timings']['tuning_provider_error_s'],3)
            self.assertEqual(row['timings']['tune_total_s'],40)
            self.assertEqual(row['tokens']['tuning'],14)
            self.assertEqual(row['tokens_unreported_calls']['tuning'],1)
            attempt=row['failed_tune_attempts'][0]
            self.assertEqual((attempt['call'],attempt['llm_calls'],attempt['partial_accepted']),(4,3,2))
            self.assertIsNone(attempt['estimates']);self.assertIsNone(attempt['after_boots'])
            self.assertEqual(row['tunes'],[]);self.assertFalse(row['post_freeze_acceptance_passed'])

    def test_complete_trace_takes_precedence_over_recovered_round(self):
        with TemporaryDirectory() as folder:
            session=fixture(Path(folder))
            h=dict(phase='llm_round',round=1,candidates=[],finish_reason='length',menu=[],menu_size=0)
            (session/'agent_trace.jsonl').write_text(json.dumps(dict(iteration=4,tuned=dict(history=[h])))+'\n')
            rows=entry.selector_rounds(session)
            self.assertEqual(rows[0]['record_source'],'trace')
            self.assertEqual(rows[0]['history']['finish_reason'],'length')
            self.assertEqual(rows[0]['history']['candidates'],[])

if __name__=='__main__':unittest.main()
