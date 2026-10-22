import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import check_hecate_libraries as gate


class HecateLibraryGateTests(unittest.TestCase):
    def test_loading_does_not_call_vm_or_frontend(self):
        symbols = {name: Mock() for name in
                   ("init", "createFunc", "save", "finalize", "initFullVM", "load", "encrypt", "decrypt")}
        library = SimpleNamespace(**symbols)
        output = io.StringIO()
        with patch.object(gate.ctypes, "CDLL", return_value=library) as load:
            with patch("sys.argv", ["check_hecate_libraries.py", "/unused/build"]):
                with contextlib.redirect_stdout(output):
                    gate.main()
        self.assertEqual(load.call_count, 2)
        for symbol in symbols.values():
            symbol.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertEqual(result["scope"], "native_library_loading_only")
        self.assertFalse(result["hecate_python_import_validated"])
        self.assertFalse(result["encrypted_execution_validated"])

    def test_loader_failure_is_not_success(self):
        output = io.StringIO()
        with patch.object(gate.ctypes, "CDLL", side_effect=OSError("missing dependency")):
            with patch("sys.argv", ["check_hecate_libraries.py", "/unused/build"]):
                with contextlib.redirect_stdout(output), self.assertRaises(OSError):
                    gate.main()
        self.assertEqual(output.getvalue(), "")

    def test_missing_symbol_is_not_success(self):
        output = io.StringIO()
        with patch.object(gate.ctypes, "CDLL", return_value=SimpleNamespace()):
            with patch("sys.argv", ["check_hecate_libraries.py", "/unused/build"]):
                with contextlib.redirect_stdout(output), self.assertRaises(AttributeError):
                    gate.main()
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
