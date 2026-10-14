"""Exact metadata authority, real analyzer boundary and malformed catalogs."""
import copy,json,tempfile,unittest
from pathlib import Path
from application_identity import source_catalog,source_closure,CATALOG,ROOTS,DYNAMIC,OPTIONAL_ROOTS,identity_manifest
from test_application_identity import backend
from benchmarks.tools.audit_application_dependencies import audit

class SourceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.base=self.root/"scripts/baseline";self.base.mkdir(parents=True)
        self.catalog=source_catalog(Path(__file__).parent)
        (self.base/CATALOG).write_text(json.dumps(self.catalog))
        names=set().union(*(set(v) for v in ROOTS.values()))
        names|={"application_identity.py","application_backend.py"}
        for name in names:(self.base/name).write_text("VALUE=1\n")
    def write(self,value):(self.base/CATALOG).write_text(json.dumps(value))
    def test_exact_previous_roots_and_dynamic_authorities(self):
        from application_identity import _catalog_values
        roots,optional,dynamic=_catalog_values(source_catalog(self.base))
        self.assertEqual(roots,ROOTS);self.assertEqual(dynamic,DYNAMIC)
        self.assertEqual(optional,OPTIONAL_ROOTS)
    def test_actual_analyzer_does_not_follow_stage_filename_metadata(self):
        (self.base/"application_identity.py").write_text(Path(__file__).parent.joinpath("application_identity.py").read_text())
        roots=["application_identity.py","candidate_trace.py"]
        closure=source_closure(self.base,roots)
        self.assertIn(CATALOG,closure["sources"])
        self.assertNotIn("deepseek_provider.py",closure["sources"])
        self.assertNotIn("application_component.py",closure["sources"])
        # An actual import must still bind the same file.
        (self.base/"candidate_trace.py").write_text("import deepseek_provider\n")
        self.assertIn("deepseek_provider.py",source_closure(self.base,roots)["sources"])
    def test_opaque_json_catalog_cannot_hide_selected_stage_roots(self):
        value=identity_manifest(self.root,backend())
        for stage,roots in ROOTS.items():
            self.assertTrue(set(roots)<=set(value[stage]["sources"]))
    def test_catalog_changes_invalidate_all_stage_configuration(self):
        first=identity_manifest(self.root,backend())
        self.catalog["dynamic_loads"][0]["authority"]+="; reviewed"
        self.write(self.catalog);second=identity_manifest(self.root,backend())
        for stage in ROOTS:self.assertNotEqual(first[stage]["sha256"],second[stage]["sha256"])
    def test_malformed_fields_paths_and_authorities_fail_closed(self):
        edits=[lambda v:v.update(extra=True),lambda v:v["roots"].pop("execution"),
            lambda v:v["roots"]["execution"].append("../escape.py"),
            lambda v:v["roots"]["execution"].append({"path":"bad.py"}),
            lambda v:v["roots"].update(execution=[]),
            lambda v:v["dynamic_loads"].append(v["dynamic_loads"][0]),
            lambda v:v["dynamic_loads"][0].update(expression="name"),
            lambda v:v["dynamic_loads"][0].update(module="/absolute.py"),
            lambda v:v["optional_roots"].update(unknown=["unknown.py"])]
        for edit in edits:
            value=copy.deepcopy(self.catalog);edit(value);self.write(value)
            with self.subTest(edit=edit),self.assertRaises(ValueError):source_catalog(self.base)
    def test_duplicate_json_fields_rejected(self):
        (self.base/CATALOG).write_text('{"format":"a","format":"b"}')
        with self.assertRaisesRegex(ValueError,"Duplicate"):source_catalog(self.base)
    def test_symlink_catalog_is_rejected(self):
        outside=self.root/"catalog.json";outside.write_text(json.dumps(self.catalog))
        (self.base/CATALOG).unlink();(self.base/CATALOG).symlink_to(outside)
        with self.assertRaisesRegex(ValueError,"symlink"):source_catalog(self.base)
    def test_developer_report_contains_causal_paths_not_execution_claims(self):
        (self.base/"candidate_trace.py").write_text("import deepseek_provider\n")
        value=audit(self.root)
        self.assertEqual(value["paid_calls"],0);self.assertFalse(value["runtime_execution"])
        self.assertEqual(value["stages"]["compilation"]["target_paths"]["deepseek_provider.py"],
                         ["candidate_trace.py","deepseek_provider.py"])
        self.assertIn("not stage independence",value["scope"])

if __name__=="__main__":unittest.main()
