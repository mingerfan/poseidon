"""Admission for the newly requested native Orion outcome and serial module."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from pt_agent.experiments import a_group_runs as records
from pt_agent.experiments import autonomous_serial_batch as serial


class OrionRecordAdmission(unittest.TestCase):
    def test_no_placement_is_a_recorded_outcome_not_a_success(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'runs.jsonl'
            row = records.empty_record(run_id='native-orion-no-plan', task='S1-Orion',
                model='resnet20', system='orion', param_group='ours-N16-boot5',
                status='no_placement', status_detail='Official solver reported no path',
                source='/original/orion/result.json')
            self.assertTrue(records.append_record(path, row))
            self.assertFalse(records.append_record(path, row))
            observed = json.loads(path.read_text())
            self.assertEqual(observed['status'], 'no_placement')
            self.assertIsNone(observed['boots_compiled'])
            self.assertIsNone(observed['pred_latency_s'])

    def test_only_the_named_new_driver_is_admitted(self):
        output = Path('/tmp/new-orion-admission')
        command = [sys.executable, '-B', '-m', serial.PREFIX+'s1_orion_compile',
                   '--out', str(output)]
        serial.check_command(command, output)
        command[3] = serial.PREFIX+'unlisted_orion_override'
        with self.assertRaises(ValueError):
            serial.check_command(command, output)
