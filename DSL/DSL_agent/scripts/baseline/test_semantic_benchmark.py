"""Small deterministic tests for graph validity, references and release integrity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from benchmark_graph import validate, signature, samples
from benchmark_suite import Builder, apply, generate, summary
from benchmark_bridge import from_legacy, to_legacy

class GraphTests(unittest.TestCase):
    def graph(self):
        b=Builder([(4,)]);return b.finish(apply(b,"input0","linear",2))
    def test_names_weights_do_not_inflate_counts(self):
        g=self.graph();h=copy.deepcopy(g);h["id"]="another"
        h["constants"]["c0"][0][0]+=0.125
        self.assertEqual(signature(g),signature(h))
        h["nodes"][0]["outputs"]=["renamed"];h["outputs"][0]["value"]="renamed"
        self.assertEqual(signature(g),signature(h))
    def test_independent_order_and_dead_nodes_do_not_inflate_counts(self):
        b=Builder([(2,)])
        a=b.node("square",["input0"]);c=b.node("negate",["input0"])
        g=b.finish(b.node("add",[a,c]));h=copy.deepcopy(g)
        h["nodes"][:2]=reversed(h["nodes"][:2])
        self.assertEqual(signature(g),signature(h))
        self.assertEqual(signature(g,True),signature(h,True))
    def test_invalid_arity_is_rejected(self):
        for op in ("add","batch_norm","linear"):
            b=Builder([(2,)])
            g=b.finish(b.node("square",["input0"]))
            g["nodes"][0].update(op=op,attrs={"eps":1e-5} if op=="batch_norm" else {})
            with self.subTest(op=op),self.assertRaises(ValueError):validate(g)
    def test_shape_changes_are_not_new_topologies(self):
        graphs=[]
        for n in (3,4):
            b=Builder([(n,)]);graphs.append(b.finish(apply(b,"input0","square")))
        self.assertNotEqual(signature(graphs[0]),signature(graphs[1]))
        self.assertEqual(signature(graphs[0],True),signature(graphs[1],True))
    def test_bad_public_constants(self):
        for value in (float("nan"),float("inf"),True,1025,[[1],[1,2]]):
            g=self.graph();g["constants"]["bad"]=value
            with self.subTest(value=value),self.assertRaises(ValueError):validate(g)
    def test_cycles_and_forward_references(self):
        g=self.graph();g["nodes"][0]["inputs"][0]=g["nodes"][0]["outputs"][0]
        with self.assertRaises(ValueError):validate(g)
    def test_output_and_input_budget(self):
        for key in ("inputs","outputs"):
            g=self.graph();g[key]=g[key]*5
            with self.subTest(key=key),self.assertRaises(ValueError):validate(g)
    def test_unknown_operator_extra_field(self):
        for change in ("op","extra"):
            g=self.graph()
            if change=="op":g["nodes"][0]["op"]="exec"
            else:g["nodes"][0]["attrs"]["unused"]=0
            with self.assertRaises(ValueError):validate(g)
    def test_split_multioutputs(self):
        b=Builder([(6,)]);parts=b.node("split",["input0"],axis=0,sections=[2,4])
        g=b.finish(*parts);self.assertEqual(validate(g)["output_shapes"],{"output0":[2],"output1":[4]})
    def test_cipher_broadcast_and_public_nonexpansion(self):
        b=Builder([(2,3),(3,)]);g=b.finish(b.node("add",["input0","input1"]))
        self.assertEqual(validate(g)["output_shapes"]["output0"],[2,3])
        b=Builder([(3,)]);y=b.node("add",["input0",b.const([[1,2,3],[4,5,6]])])
        with self.assertRaises(ValueError):b.finish(y)
    def test_legacy_roundtrip(self):
        g=self.graph();h=from_legacy(to_legacy(g))
        self.assertEqual(signature(g),signature(h))
    def test_negative_bn_variance(self):
        b=Builder([(1,2)]);x=apply(b,"input0","batch_norm");g=b.finish(x)
        g["constants"]["c1"]=[-1,1]
        with self.assertRaises(ValueError):validate(g)

class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=generate()
    def test_counts_and_categories(self):
        s=summary(self.rows)
        self.assertEqual(s["models"],1200);self.assertEqual(s["unique_signatures"],1200)
        self.assertGreaterEqual(s["topology_groups"],300);self.assertGreaterEqual(s["small_models"],960)
        self.assertEqual(s["split_counts"],{"development":720,"validation":240,"holdout":240})
    def test_boundary_extremes_are_present(self):
        import math
        from benchmark_graph import validate
        graphs=[r["model"] for r in self.rows]
        self.assertEqual(max(len(g["outputs"]) for g in graphs),4)
        self.assertEqual(max(sum(math.prod(i["shape"]) for i in g["inputs"]) for g in graphs),256)
        self.assertEqual(max(sum(math.prod(x) for x in validate(g)["output_shapes"].values()) for g in graphs),256)
        sizes={math.prod(i["shape"]) for g in graphs for i in g["inputs"]}
        self.assertTrue({1,127,128,129,255,256}.issubset(sizes))
    def test_split_separation(self):
        groups={}
        for r in self.rows:groups.setdefault(r["topology"],set()).add(r["split"])
        self.assertTrue(all(len(x)==1 for x in groups.values()))
    def test_all_registered_ops_have_fixtures(self):
        from benchmark_graph import OPS
        observed={n["op"] for r in self.rows for n in r["model"]["nodes"]}
        self.assertEqual(observed,set(OPS))
    def test_helper_denominator(self):
        from benchmark_suite import HELPERS
        from collections import Counter
        self.assertEqual(Counter(r["metadata"].get("helper") for r in self.rows if r["category"]=="helper"),
                         Counter({h:8 for h in HELPERS}))
    def test_no_legacy_duplicates(self):
        from benchmark_semantics import legacy_overlap
        d=legacy_overlap(self.rows);self.assertEqual(d["overlaps"],[]);self.assertEqual(d["normalization_errors"],[])
    def test_directed_contexts_are_only_planned(self):
        from benchmark_semantics import ledger
        d=ledger(self.rows);by_id={r["model"]["id"]:r for r in self.rows}
        groups={}
        from unified_graph_exercises import spec
        for t in d["directed_tasks"]:
            if t["exercise"]:
                if t["profile"]=="hecate-unified-public-v1":
                    from unified_public_exercises import spec as public_spec
                    self.assertEqual(t["instruction"],public_spec(t["exercise"])["instruction"])
                else:self.assertEqual(t["instruction"],spec(t["exercise"])["instruction"])
            self.assertEqual(t["execution_status"],"not_run")
            groups.setdefault(t["requirement"],set()).add(by_id[t["model_id"]]["topology"])
        self.assertTrue(all(len(x)>=3 for x in groups.values()))
        self.assertEqual(d["evidence_counts"]["agent_cases"],0)
    def test_public_request_has_no_answers(self):
        # Source model data does not contain private probes or reference arrays.
        for r in self.rows:self.assertEqual(set(r["model"]),{"format","id","inputs","constants","nodes","outputs"})
    def test_frozen_release(self):
        from benchmark_runner import load,DEFAULT
        rows,_=load(DEFAULT);self.assertEqual([r["model"] for r in rows],[r["model"] for r in self.rows])

class ReferenceTests(unittest.TestCase):
    def test_hand_computed_linear(self):
        import numpy as np
        from benchmark_math import evaluate
        from benchmark_torch import evaluate as torch_eval
        b=Builder([(2,)]);g=b.finish(b.node("linear",["input0",b.const([[2.,-3.]]),b.const([0.5])]))
        inputs={"input0":np.array([4.,5.])}
        for fn in (evaluate,torch_eval):np.testing.assert_array_equal(fn(g,inputs)["output0"],[-6.5])
    def test_spatial_and_slice_sensitive_inputs(self):
        import numpy as np
        from benchmark_math import evaluate
        from benchmark_torch import evaluate as torch_eval
        from benchmark_suite import spatial
        for variant in range(4):
            b=Builder([(1,2,3,3)]);x=spatial(b,"input0",2,variant);g=b.finish(x)
            for inp in samples(g,4):
                np.testing.assert_allclose(evaluate(g,inp)["output0"],torch_eval(g,inp)["output0"],atol=1e-12,rtol=1e-12)
    def test_split_swapped_outputs_detected(self):
        import numpy as np
        from benchmark_math import evaluate
        b=Builder([(4,)]);parts=b.node("split",["input0"],axis=0,sections=[2,2]);g=b.finish(*parts)
        y=evaluate(g,{"input0":np.array([1.,2.,3.,4.])})
        self.assertFalse(np.array_equal(y["output0"],y["output1"]))
    def test_probe_generation_deterministic(self):
        import numpy as np
        b=Builder([(3,),(2,)]);g=b.finish(b.node("mean",["input0"],axes=[0],keepdims=False))
        for a,b in zip(samples(g),samples(g)):
            for k in a:np.testing.assert_array_equal(a[k],b[k])

if __name__=="__main__":unittest.main()
