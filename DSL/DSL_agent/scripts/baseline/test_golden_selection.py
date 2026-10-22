"""Case selection only; no tracing or encrypted-execution claim."""
import contextlib
import io
import unittest
from seal_cpu_golden import CASES, EXTENDED_CASES, parse_args, selected_cases


class GoldenSelectionTests(unittest.TestCase):
    def test_single_mlp_does_not_expand_to_extended_suite(self):
        self.assertEqual(selected_cases(parse_args(["--case", "mlp4x4x2"])), ("mlp4x4x2",))

    def test_existing_suite_defaults_are_preserved(self):
        self.assertEqual(selected_cases(parse_args([])), CASES)
        self.assertEqual(selected_cases(parse_args(["--suite", "extended"])), EXTENDED_CASES)

    def test_case_and_suite_cannot_be_combined(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parse_args(["--case", "mlp4x4x2", "--suite", "extended"])
        self.assertEqual(error.exception.code, 2)

    def test_unknown_case_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parse_args(["--case", "arbitrary-code.py"])
        self.assertEqual(error.exception.code, 2)
