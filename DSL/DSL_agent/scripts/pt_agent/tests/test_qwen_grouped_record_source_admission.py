"""New explicit replay source admission; no candidate or model is evaluated."""
import json
from pathlib import Path
import tempfile
import unittest

from pt_agent.experiments import autonomous_serial_batch as serial
from pt_agent.experiments import qwen_agent_grouped_record as lifetime
from pt_agent.experiments import qwen_prefill_compile as frozen


class GroupedRecordSourceAdmission(unittest.TestCase):
    def command(self, case, module):
        return ['python3', '-B', '-m', module, '--out', str(case),
                '--partial-from', str(case.parent/'partial'),
                '--resume-from', str(case.parent/'resume'), '--worker-phase', 'prompt',
                '--session', str(case/'session')]

    def test_saved_new_worker_is_parsed_without_running_the_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)/'new-case'; case.mkdir()
            (case/'command-1.json').write_text(json.dumps(self.command(case, lifetime.MODULE)))
            entry, args = frozen.source_arguments(case)
            self.assertIs(entry, lifetime)
            self.assertIsNone(args.out)
            self.assertEqual(args.worker_phase, 'prompt')
            self.assertEqual(Path(args.session), case/'session')
            self.assertEqual((args.memory_gib, args.rss_gib, args.minimum_available_gib), (24, 16, 1.25))

    def test_unknown_similar_module_remains_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)/'new-case'; case.mkdir()
            (case/'command-1.json').write_text(json.dumps(self.command(case, lifetime.MODULE+'_unreviewed')))
            with self.assertRaisesRegex(ValueError, 'trusted Qwen Agent worker'):
                frozen.source_arguments(case)

    def test_serial_admits_only_current_default_parent_resources(self):
        out = '/tmp/qwen-grouped-record-new-case'
        command = ['python3', '-B', '-m', lifetime.MODULE, '--out', out, '--through', 'agent']
        self.assertEqual(serial.check_command(command, out), command)
        self.assertEqual(serial.parent_limits({'argv': command}), (24, 16))
        with self.assertRaises(ValueError):
            serial.parent_limits({'argv': command, 'outer_address_space_gib': 32})


if __name__ == '__main__':
    unittest.main()
