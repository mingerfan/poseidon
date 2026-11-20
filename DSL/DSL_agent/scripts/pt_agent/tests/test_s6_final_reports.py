import json
from pathlib import Path
import tempfile
import unittest

from pt_agent.experiments import s6_final_reports as final


class FinalReportTests(unittest.TestCase):
    def test_optional_metadata_admission_has_no_local_or_remote_call(self):
        with tempfile.TemporaryDirectory() as root:
            case = Path(root)
            result = case / 'result.json'
            result.write_text(json.dumps(dict(metadata_only=True, scientific_attempts=0,
                reason='Insufficient remaining time for the optional full budget')))
            row = final.enrich(dict(source=str(result), model='bert_tiny', status='driver_error'))
            self.assertEqual(row['status'], 'not_run')
            self.assertEqual(row['offline_replies'], 0)
            self.assertEqual(row['real_API_calls'], 0)
            self.assertFalse(row['paper_eligible'])

    def test_native_and_transformer_report_percentages_keep_original_units(self):
        cnn = final.report_metrics(dict(status='ok', report_samples_completed=8000,
            reference_top1_percent=92.5, approximate_top1_percent=92.45,
            accuracy_drop_pp=.05, report_policy_passed=True))
        bert = final.report_metrics(dict(status='ok', report_samples=872, worker=dict(
            validation_metrics=dict(top1=82.9, exact_top1=82.9, acc_drop_pp=0.), task_policy_passed=True)))
        self.assertEqual(cnn['frozen_top1_percent'], 92.45)
        self.assertEqual(bert['frozen_top1_percent'], 82.9)
        self.assertEqual(bert['samples'], 872)

    def test_heldout_uses_exact_program_and_split_without_confirmation_feedback(self):
        with tempfile.TemporaryDirectory() as root:
            case = Path(root)
            session = case / 'work/agent'
            session.mkdir(parents=True)
            text = 'def model(x):\n return x\n'
            (session / 'program.ptdsl').write_text(text)
            rows = [dict(program=text, split='heldout', metrics=dict(top1=92.4, total=1000)),
                    dict(program=text, split='confirm', metrics=dict(top1=90.7, total=1000)),
                    dict(program='def model(x):\n return -x\n', split='heldout', metrics=dict(top1=0., total=1000))]
            (session / 'observed-quality.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
            measured = final.heldout_metrics(case, 'resnet20')
            self.assertEqual(measured['top1'], 92.4)
            self.assertEqual(measured['extra_model_evaluations'], 0)

    def test_accuracy_derivation_requires_all1000_reference_predictions(self):
        with tempfile.TemporaryDirectory() as root:
            case = Path(root)
            session = case / 'work/agent'
            session.mkdir(parents=True)
            text = 'def model(x):\n return x\n'
            (session / 'program.ptdsl').write_text(text)
            row = dict(fb=dict(validation=dict(exact_top1=92.4)),
                       tuned=dict(source=text, history=[dict(phase='result', agree=1000, total=1000)]))
            trace = session / 'agent_trace.jsonl'
            trace.write_text(json.dumps(row)+'\n')
            self.assertEqual(final.heldout_metrics(case, 'resnet20')['top1'], 92.4)
            row['tuned']['history'][0]['agree'] = 999
            trace.write_text(json.dumps(row)+'\n')
            self.assertFalse(final.heldout_metrics(case, 'resnet20')['available'])

    def test_selected_rule_reply_uses_its_recorded_original_validation(self):
        with tempfile.TemporaryDirectory() as root:
            case = Path(root)
            session = case / 'work/agent'
            session.mkdir(parents=True)
            text = 'def model(x):\n return x\n'
            (session / 'program.ptdsl').write_text(text)
            state = dict(confirmation=dict(selected=dict(origin='llm', iteration=1, source=text)))
            (session / 'SESSION.json').write_text(json.dumps(state))
            row = dict(iteration=1, fb=dict(validation=dict(top1=93.8, exact_top1=93.9,
                agree=998, total=1000, acc_drop_pp=.1)))
            (session / 'agent_trace.jsonl').write_text(json.dumps(row)+'\n')
            self.assertEqual(final.heldout_metrics(case, 'vgg16')['top1'], 93.8)
            state['confirmation']['selected']['source'] = 'def model(x):\n return -x\n'
            (session / 'SESSION.json').write_text(json.dumps(state))
            self.assertFalse(final.heldout_metrics(case, 'vgg16')['available'])


if __name__ == '__main__':
    unittest.main()
