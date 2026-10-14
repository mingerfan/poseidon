"""Qualification subsets cannot silently omit or substitute named cases."""
import importlib.util
import unittest
from pathlib import Path

path=Path(__file__).parent/"benchmarks/tools/qualify_agent_component.py"
spec=importlib.util.spec_from_file_location("component_qualification_tool",path)
tool=importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

class QualificationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.fixtures=[dict(id="a"),dict(id="b"),dict(id="c")]

    def test_no_filter_retains_all(self):
        self.assertIs(tool.select_fixtures(self.fixtures,[]),self.fixtures)

    def test_filter_preserves_frozen_order(self):
        self.assertEqual(tool.select_fixtures(self.fixtures,["c","a"]),
                         [self.fixtures[0],self.fixtures[2]])

    def test_unknown_is_rejected(self):
        with self.assertRaises(ValueError):
            tool.select_fixtures(self.fixtures,["a","missing"])

    def test_duplicates_are_rejected(self):
        with self.assertRaises(ValueError):
            tool.select_fixtures(self.fixtures,["a","a"])

if __name__=="__main__": unittest.main()
