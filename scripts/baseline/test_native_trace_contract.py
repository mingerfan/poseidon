"""Offline evidence-file protocol checks, not native or encrypted evidence."""
import ast
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import candidate_trace


class TraceEvidenceTests(unittest.TestCase):
    def test_both_paths_use_shared_save(self):
        tree = ast.parse(Path(candidate_trace.__file__).read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'save_trace']
        self.assertEqual({n.args[2].value for n in calls}, {
            'validated_native_AST_to_Hecate_functions', 'validated_AST_to_Hecate_objects'})
        self.assertEqual(len(calls), 2)

    def test_protocol_and_no_success_evidence_when_save_fails(self):
        hc = Mock()
        with patch.object(candidate_trace, 'Path') as path:
            candidate_trace.save_trace(hc, {'request_id': 'frozen-id'}, 'native-test')
            hc.save.assert_called_once_with('/out', '/out')
            path.assert_called_once_with('/out/trace-evidence.json')
            evidence = json.loads(path.return_value.write_text.call_args.args[0])
            self.assertEqual(evidence, dict(frontend='real_Hecate',
                candidate_python_executed=False, construction='native-test', request_id='frozen-id'))
        hc.save.side_effect = RuntimeError('save failed')
        with patch.object(candidate_trace, 'Path') as path, self.assertRaises(RuntimeError):
            candidate_trace.save_trace(hc, {'request_id': 'frozen-id'}, 'native-test')
        path.assert_not_called()


if __name__ == '__main__':
    unittest.main()
