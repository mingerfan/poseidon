from pathlib import Path
from types import SimpleNamespace
import json
import os
import tempfile
import unittest

from pt_agent.experiments.s2_heir_serial import closed_case, command, same_birth


class SerialAdmissionTests(unittest.TestCase):
    def test_predecessor_birth_is_exact(self):
        text = Path('/proc/self/stat').read_text()
        birth = text[text.rfind(')')+2:].split()[19]
        self.assertTrue(same_birth(os.getpid(), birth))
        self.assertFalse(same_birth(os.getpid(), str(int(birth)+1)))

    def test_completion_and_closed_descendants_are_required(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.assertFalse(closed_case(root))
            (root/'completion.json').write_text(json.dumps(dict(registry_persisted=True, required_artifacts_persisted=True)))
            (root/'paper-record.json').write_text('{}')
            result = dict(monitor_stopped=True, observation=dict(remaining_pids=[]))
            (root/'result.json').write_text(json.dumps(result))
            self.assertTrue(closed_case(root))
            result['observation']['remaining_pids'] = [123]
            (root/'result.json').write_text(json.dumps(result))
            self.assertFalse(closed_case(root))

    def test_successors_keep_original_inputs_and_same_driver(self):
        args = SimpleNamespace(source_check=Path('/sources.json'), parameter_probe=Path('/params.json'),
                               base_pipeline=Path('/pipeline.txt'), paper_runs=Path('/runs.jsonl'))
        for model in ('vgg16', 'alexnet'):
            case, argv = command(args, model)
            self.assertIn('pt_agent.experiments.s2_heir_compile', argv)
            self.assertEqual(argv[argv.index('--model')+1], model)
            self.assertEqual(argv[argv.index('--source-check')+1], '/sources.json')
            self.assertIn(model, case.name)


if __name__ == '__main__':
    unittest.main()
