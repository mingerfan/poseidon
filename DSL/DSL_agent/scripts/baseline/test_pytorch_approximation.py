"""Selection conflicts, scope and provenance are deterministic data checks."""
import copy,unittest
from benchmark_graph import digest
from pytorch_approximation import FORMAT,Resolver,validate,check_record
P=dict(method="polynomial",basis="power",domain=[-2.,2.],coefficients=[0.,.5,.125])
def config(defaults=None,modules=None,nodes=None):
    return dict(format=FORMAT,defaults=defaults or {},modules=modules or [],nodes=nodes or [])
def mod(path,p=P):return dict(path=path,function="relu",profile=p)
def node(name,p=P):return dict(name=name,function="relu",profile=p)

class SelectionTests(unittest.TestCase):
    def test_three_priorities_and_shadowed_valid_defaults(self):
        a=P;b=dict(P,coefficients=[0.,.25]);c=dict(P,coefficients=[0.,.125])
        r=Resolver(config(dict(relu=a),[mod("act",b)],[node("act_1",c)]))
        self.assertEqual(r.select("act","relu","relu",["act"]),b)
        self.assertEqual(r.select("act_1","relu","relu",["act"]),c)
        self.assertEqual(r.select("relu","relu","relu",[]),a)
        report=r.finish()
        self.assertEqual([v["source"]["kind"] for v in report["selections"]],["modules","nodes","defaults"])
    def test_duplicate_selector_same_priority_rejected(self):
        for field,items in [("nodes",[node("relu"),node("relu")]),("modules",[mod("act"),mod("act")])]:
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"Conflicting"):
                validate(config(**{field:items}))
    def test_overlapping_module_scopes_conflict_even_with_node_override(self):
        r=Resolver(config({},[mod("block"),mod("block.act")],[node("act")]))
        with self.assertRaisesRegex(ValueError,"Conflicting"):
            r.select("act","relu","relu",["block","block.act"])
    def test_unmatched_node_module_and_default_rejected(self):
        for value in (config({},[],[node("unused")]),config({},[mod("unused")]),
                      config(dict(relu=P))):
            r=Resolver(value)
            with self.assertRaisesRegex(ValueError,"Unmatched"):r.finish()
    def test_missing_profile_rejected_at_public_node(self):
        with self.assertRaisesRegex(ValueError,"Explicit approximation.*relu.*act"):
            Resolver(config()).select("act","relu","relu",[])
    def test_different_composite_functions_at_same_node_are_valid(self):
        e=dict(method="taylor",domain=[-2.,2.],degree=2)
        rec=dict(method="goldschmidt",domain=[.5,8.],iterations=1)
        r=Resolver(config({},[],[dict(name="sm",function="exp",profile=e),
                                dict(name="sm",function="reciprocal",profile=rec)]))
        self.assertEqual(r.select("sm","softmax","exp",[]),e)
        self.assertEqual(r.select("sm","softmax","reciprocal",[]),rec)
        self.assertEqual(len(r.finish()["selections"]),2)
    def test_source_profiles_are_snapshots(self):
        value=config(dict(relu=copy.deepcopy(P)));r=Resolver(value)
        value["defaults"]["relu"]["coefficients"][1]=9.
        selected=r.select("act","relu","relu",[])
        selected["coefficients"][1]=8.
        self.assertEqual(r.finish()["selections"][0]["profile_sha256"],digest(P))
    def test_exact_module_path_no_glob_or_prefix(self):
        r=Resolver(config({},[mod("block")]))
        with self.assertRaises(ValueError):r.select("act","relu","relu",["block2"])
        for path in ("block.*","../block","","a/b"):
            with self.subTest(path=path),self.assertRaises(ValueError):validate(config({},[mod(path)]))
    def test_malformed_data_and_profiles_rejected(self):
        changes=[
            lambda v:v.update(format="unknown"),
            lambda v:v.update(callback="run"),
            lambda v:v["defaults"].update(unknown=P),
            lambda v:v["defaults"]["relu"].update(coefficients=[True]),
            lambda v:v["defaults"]["relu"].update(domain=[2.,-2.]),
            lambda v:v["defaults"]["relu"].update(coefficients=[float("nan")]),
            lambda v:v.update(modules={}),
            lambda v:v.update(nodes=[dict(name="x",function="rsqrt",profile=P)]),
        ]
        for change in changes:
            bad=config(dict(relu=copy.deepcopy(P)));change(bad)
            with self.subTest(change=change),self.assertRaises(ValueError):validate(bad)
    def test_selector_and_byte_limits(self):
        with self.assertRaises(ValueError):validate(config({},[],[node("x"+str(i)) for i in range(129)]))
        with self.assertRaises(ValueError):validate(config({},[mod("x"*257)]))
        with self.assertRaises(ValueError):validate(config(dict(relu=dict(P,coefficients=[1.]*129))))
    def test_bad_source_path_types_rejected(self):
        for paths in ({},[[]],["act","act"],["../act"]):
            with self.subTest(paths=paths),self.assertRaises(ValueError):
                Resolver(config(dict(relu=P))).select("act","relu","relu",paths)
    def test_graph_record_checks_exact_profiles_and_source_selection(self):
        r=Resolver(config(dict(relu=P)));r.select("act","relu","relu",[])
        record=r.finish()
        original=dict(nodes=[dict(id="fx_act",op="relu",attrs=dict(approximation=P))])
        self.assertEqual(check_record(record,original,config(dict(relu=P))),record)
        for change in (
                lambda v:v.update(configuration_sha256="0"*64),
                lambda v:v["selections"][0].update(profile_sha256="0"*64),
                lambda v:v["selections"][0].update(operator="gelu"),
                lambda v:v["selections"][0].update(node_id="fx_other"),
                lambda v:v["selections"][0].update(source=dict(kind="modules",selector=0)),
                lambda v:v["selections"].clear(),
                lambda v:v["selections"].append(copy.deepcopy(v["selections"][0]))):
            bad=copy.deepcopy(record);change(bad)
            with self.subTest(change=change),self.assertRaises(ValueError):
                check_record(bad,original,config(dict(relu=P)))
        bad=copy.deepcopy(original);bad["nodes"][0]["attrs"]["approximation"]["coefficients"][1]=.25
        with self.assertRaisesRegex(ValueError,"differs"):check_record(record,bad,config(dict(relu=P)))

if __name__=="__main__":unittest.main()
