"""Saved-command adapter and C2 admission; no model, provider or compiler."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_prefill_compile as subject


class GuardedPrefillAdmission(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.case = self.root/'guarded'
        (self.case/'session').mkdir(parents=True)
        self.argv = ['python3', '-B', '-m', subject.guarded.MODULE,
            '--worker-phase', 'calibration', '--session', str(self.case/'session'),
            '--resume-from', str(self.root/'original-nine'), '--partial-from',
            str(self.root/'failed-bounded'), '--model-info', str(self.root/'info'),
            '--model', str(self.root/'trained-full24'), '--calibration',
            str(self.case/'calibration.json'), '--profile',
            str(subject.paths.PROFILES_DIR/'qwen48/compiler-config.json'),
            '--frontend', str(self.root/'expr.py'), '--metadata-template',
            str(self.root/'template.json')]
        self.save('command-1.json', self.argv)

    def save(self, relative, value):
        path = self.case/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)+'\n')

    def frozen_fixture(self):
        source = dict(original_layers=24, graph_operation_counts=dict(softmax=24, shuffle=48),
                      guarded_replay=dict(committed_calls=12, pending_provider_attempt=13))
        self.save('result.json', dict(round=subject.guarded.ROUND, mode=subject.guarded.MODE,
            model=subject.MODEL, status='ok', candidate_frozen=True, source=source))
        self.save('completion.json', dict(status='ok', registry_persisted=True,
                                        required_artifacts_persisted=True))
        self.save('session/SESSION.json', dict(failed=None, calls=15))
        self.save('calibration.json', {})
        self.save('unit-width-bridge.json', {})
        outcome = dict(done=True, frozen=True, status='ok', selected=dict(iteration=14, origin='llm'))
        return source, outcome

    def test_natural_calibration_command_uses_real_guarded_parser_without_running_it(self):
        with patch.object(subject.guarded, 'main', side_effect=AssertionError('Must not execute')):
            entry, args = subject.source_arguments(self.case)
        self.assertIs(entry, subject.guarded)
        self.assertEqual(args.partial_from, self.root/'failed-bounded')
        self.assertEqual(args.resume_from, self.root/'original-nine')
        self.assertEqual((args.memory_gib, args.rss_gib, args.minimum_available_gib), (24, 16, 1.25))
        self.assertEqual(args.session, self.case/'session')
        self.assertIsNone(args.out)

    def test_partial_source_remains_a_required_original_parser_argument(self):
        argv = list(self.argv); index = argv.index('--partial-from'); del argv[index:index+2]
        self.save('command-1.json', argv)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            subject.source_arguments(self.case)

    def test_step_command_or_foreign_session_is_refused(self):
        argv = list(self.argv); argv[argv.index('--worker-phase')+1] = 'step'
        self.save('command-1.json', argv)
        with self.assertRaisesRegex(ValueError, 'original session'):
            subject.source_arguments(self.case)
        argv = list(self.argv); argv[argv.index('--session')+1] = str(self.root/'other-session')
        self.save('command-1.json', argv)
        with self.assertRaisesRegex(ValueError, 'original session'):
            subject.source_arguments(self.case)

    def test_only_independently_frozen_guarded_source_is_admitted(self):
        source, outcome = self.frozen_fixture()
        with patch.object(subject.guarded, 'session_outcome', return_value=outcome), \
                patch.object(subject.guarded, 'source_gate', return_value=source):
            result = subject.frozen_source(self.case)
        self.assertEqual(result['entry_module'], subject.guarded.MODULE)
        self.assertEqual((result['full_model_layers'], result['inherited_calls'], result['selected_iteration']),
                         (24, 15, 14))
        self.assertFalse(result['report_outcome_used_for_selection'])
        for changes in (dict(done=False), dict(frozen=False), dict(status='quality_rejected')):
            with self.subTest(changes=changes), \
                    patch.object(subject.guarded, 'session_outcome', return_value=dict(outcome, **changes)):
                with self.assertRaisesRegex(ValueError, 'C2-confirmed'):
                    subject.frozen_source(self.case)

    def test_old_resource_error_cannot_be_relabelled_as_a_frozen_source(self):
        source, outcome = self.frozen_fixture()
        self.save('result.json', dict(round=subject.guarded.ROUND, mode=subject.guarded.MODE,
            model=subject.MODEL, status='error', candidate_frozen=False, source=source))
        with patch.object(subject.guarded, 'session_outcome', return_value=outcome):
            with self.assertRaisesRegex(ValueError, 'C2-confirmed'):
                subject.frozen_source(self.case)


if __name__ == '__main__':
    unittest.main()
