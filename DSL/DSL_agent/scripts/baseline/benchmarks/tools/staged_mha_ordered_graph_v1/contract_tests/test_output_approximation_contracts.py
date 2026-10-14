"""Standard-library compatibility and rejection controls for isolated contracts.

These controls do not exercise Torch dataflow, compilation or encrypted execution.
MHA source location maps passed here represent a separately verified host result;
the source-record checker is responsible for the mathematical map authentication.
"""
import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
BASE=STAGE.parents[2]
OVERLAY=STAGE/"overlay"
sys.path.insert(0,str(BASE))
sys.path.insert(0,str(OVERLAY))
import pytorch_source_contract as outputs
import pytorch_approximation as approximations
from benchmark_graph import canonical,digest


def old_module(name):
    spec=importlib.util.spec_from_file_location("_untouched_"+name,BASE/(name+".py"))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def config():
    return {"format":approximations.FORMAT,"defaults":{
        "exp":{"method":"taylor","domain":[-.5,.5],"degree":4},
        "reciprocal":{"method":"goldschmidt","domain":[.5,8.],"iterations":4}},
        "modules":[],"nodes":[]}


def graph():
    c=config()
    return {"nodes":[
        {"id":"fx_attention_exp_a","op":"exp","attrs":{"approximation":c["defaults"]["exp"]}},
        {"id":"fx_attention_exp_b","op":"exp","attrs":{"approximation":c["defaults"]["exp"]}},
        {"id":"fx_attention_recip","op":"reciprocal","attrs":{"approximation":c["defaults"]["reciprocal"]}}]}


def locations():
    return {n["id"]:{"source_node":"attention","module_paths":["attention"]} for n in graph()["nodes"]}


def selected(model,configuration,loc=None):
    resolver=approximations.Resolver(configuration)
    for n in model["nodes"]:
        for function,expected in approximations._components(n):
            source=n["id"][3:] if loc is None else loc[n["id"]]["source_node"]
            paths=[] if loc is None else loc[n["id"]]["module_paths"]
            resolver.select(source,n["op"],function,paths)
    return resolver.finish()


class ContractControls(unittest.TestCase):
    def test_imports_isolated_contracts(self):
        self.assertEqual(Path(outputs.__file__).resolve().parent,OVERLAY)
        self.assertEqual(Path(approximations.__file__).resolve().parent,OVERLAY)

    def test_old_output_trees_and_canonical_bytes_identical(self):
        old=old_module("pytorch_source_contract")
        x=object();y=object()
        for value,names in [(x,["x"]),((x,y),["x","y"]),([x,{"y":y}],["x","y"])]:
            a,ta=old.output_tree(value,lambda v:v is x or v is y,names)
            b,tb=outputs.output_tree(value,lambda v:v is x or v is y,names)
            self.assertEqual(a,b);self.assertEqual(canonical(ta),canonical(tb));self.assertEqual(digest(ta),digest(tb))
            self.assertEqual(old.restore_outputs(dict(zip(names,range(len(names)))),ta),
                             outputs.restore_outputs(dict(zip(names,range(len(names)))),tb))

    def test_old_scalar_tree_bytes_identical(self):
        old=old_module("pytorch_source_contract");leaf=object()
        _,before=old.output_tree(leaf,lambda x:x is leaf,["mean"],scalar_tensor=lambda x:True)
        _,after=outputs.output_tree(leaf,lambda x:x is leaf,["mean"],scalar_tensor=lambda x:True)
        self.assertEqual(canonical(before),canonical(after));self.assertEqual(after,{"kind":"tensor","name":"mean","shape":[]})

    def test_static_none_explicit_and_has_no_extra_named_port(self):
        leaf=object()
        found,tree=outputs.output_tree({"result":leaf,"aux":[None]},lambda x:x is leaf,["result"],allow_none=True)
        self.assertEqual(found,[leaf])
        self.assertEqual(outputs.restore_outputs({"result":19},tree,allow_none=True),{"result":19,"aux":[None]})
        with self.assertRaises(ValueError):outputs.validate_output_tree(tree,["result"])
        with self.assertRaises(ValueError):outputs.restore_outputs({"result":19},tree)

    def test_none_default_and_nonboolean_policy_rejected(self):
        leaf=object()
        with self.assertRaises(ValueError):outputs.output_tree((leaf,None),lambda x:x is leaf,["x"])
        for policy in (1,"true",None):
            with self.assertRaises(ValueError):outputs.output_tree(leaf,lambda x:x is leaf,["x"],allow_none=policy)
            with self.assertRaises(ValueError):outputs.validate_output_tree({"kind":"tensor","name":"x"},["x"],allow_none=policy)

    def test_none_node_extra_fields_and_dynamic_leaf_rejected(self):
        for tree in ({"kind":"none","value":0},{"kind":"none","name":"ghost"}):
            with self.assertRaises(ValueError):outputs.validate_output_tree(tree,[],allow_none=True)
        leaf=object()
        with self.assertRaises(ValueError):outputs.output_tree((leaf,3),lambda x:x is leaf,["x"],allow_none=True)

    def test_none_signature_count_cannot_replace_tensor(self):
        with self.assertRaises(ValueError):outputs.validate_output_tree({"kind":"none"},["x"],allow_none=True)
        with self.assertRaises(ValueError):outputs.output_tree(None,lambda x:False,["x"],allow_none=True)

    def test_old_approximation_record_bytes_identical_and_accepted(self):
        old=old_module("pytorch_approximation");model=graph();cfg=config()
        before=selected(model,cfg);after=approximations.check_record(copy.deepcopy(before),model,cfg)
        self.assertEqual(canonical(old.check_record(copy.deepcopy(before),model,cfg)),canonical(after))

    def test_original_source_location_repeats_bind_each_expansion_component(self):
        model=graph();cfg=config();loc=locations();record=selected(model,cfg,loc)
        self.assertEqual({r["node_id"] for r in record["selections"]},{"fx_attention"})
        self.assertEqual(len(record["selections"]),3)
        self.assertEqual(approximations.check_record(record,model,cfg,source_locations=loc),record)
        with self.assertRaises(ValueError):approximations.check_record(record,model,cfg)

    def test_source_location_fields_inventory_and_noncomponent_rejected(self):
        model=graph();cfg=config();loc=locations();record=selected(model,cfg,loc)
        for bad in [dict(loc,fx_unknown={"source_node":"attention","module_paths":["attention"]}),
                    {**loc,"fx_attention_exp_a":{"source_node":"attention","module_paths":["attention"],"private":[]}},
                    {**loc,"fx_attention_exp_a":{"source_node":"not valid","module_paths":["attention"]}}]:
            with self.assertRaises(ValueError):approximations.check_record(record,model,cfg,source_locations=bad)
        model["nodes"].append({"id":"fx_view","op":"reshape","attrs":{"shape":[1]}})
        loc["fx_view"]={"source_node":"attention","module_paths":["attention"]}
        with self.assertRaises(ValueError):approximations.check_record(record,model,cfg,source_locations=loc)

    def test_resigned_source_row_or_module_path_tamper_rejected(self):
        model=graph();cfg=config();loc=locations();record=selected(model,cfg,loc)
        for key,value in (("node_id","fx_attention_exp_a"),("module_paths",["other"]),
                          ("function","reciprocal"),("operator","multiply")):
            bad=copy.deepcopy(record);bad["selections"][0][key]=value
            with self.assertRaises(ValueError):approximations.check_record(bad,model,cfg,source_locations=loc)

    def test_expanded_profile_change_cannot_reuse_old_selection(self):
        model=graph();cfg=config();loc=locations();record=selected(model,cfg,loc)
        model["nodes"][1]["attrs"]["approximation"]["degree"]=6
        with self.assertRaises(ValueError):approximations.check_record(record,model,cfg,source_locations=loc)

    def test_extra_missing_rows_are_rejected(self):
        model=graph();cfg=config();loc=locations();record=selected(model,cfg,loc)
        for rows in (record["selections"][:-1],record["selections"]+[record["selections"][0]]):
            bad=copy.deepcopy(record);bad["selections"]=rows
            with self.assertRaises(ValueError):approximations.check_record(bad,model,cfg,source_locations=loc)


if __name__=="__main__":unittest.main(verbosity=2)
