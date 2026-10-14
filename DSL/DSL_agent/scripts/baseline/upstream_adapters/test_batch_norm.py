"""Actual upstream Python on public vectors: plaintext tests, NOT FHE evidence."""
import copy
import json
import sys
import unittest
from pathlib import Path
import numpy as np
from benchmark_suite import Builder
from benchmark_math import evaluate
from benchmark_graph import samples
from benchmark_runner import ROOT
from upstream_adapters.batch_norm import bind,apply,SLOTS

class Vector:
    def __init__(self,data):self.data=np.asarray(data,dtype=np.float64)
    def __mul__(self,value):return Vector(self.data*np.asarray(value,dtype=np.float64))
    def __add__(self,value):return Vector(self.data+np.asarray(value,dtype=np.float64))

from upstream_adapters.scenarios import batch_norm_model as model

class BatchNormAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from poly_dependencies import verify,TARGET
        verify()
        import os,importlib.util
        from hecate_python_env import WORK
        from python_compiler_smoke import BUILD
        compatibility=WORK/"build-dacapo/hecate-python-root"
        if (compatibility/"build").resolve()!=BUILD or not (BUILD/"lib/libHecateFrontend.so").is_file():
            raise ValueError("Existing pinned frontend compatibility root required")
        os.environ["HECATE"]=str(compatibility)
        module=importlib.util.spec_from_file_location("bn_plain_frontend",ROOT/"third_party/dacapo/python/hecate/hecate/expr.py")
        hc=importlib.util.module_from_spec(module);sys.modules[module.name]=hc
        module.loader.exec_module(hc)
        sys.modules["hecate"]=hc
        sys.path[:0]=[str(TARGET),str(ROOT/"third_party/dacapo/python/poly")]
        import poly.Func as helpers
        import poly.MPCB as mpcb
        cls.helpers=helpers;cls.mpcb=mpcb

    def test_public_vectors_match_independent_reference_and_period(self):
        shapes=[(1,c) for c in range(2,10)]+[(1,3,2),(1,2,2,3)]
        for shape in shapes:
            g=model(shape);before=copy.deepcopy(g);spec=bind(g)
            count=spec["logical_elements"];period=max(4,spec["closure_period"])
            for inputs in samples(g,4):
                flat=inputs["input0"].reshape(-1)
                vec=Vector(np.tile(np.pad(flat,(0,period-count)),SLOTS//period))
                out,record=apply(g,vec,self.helpers,self.mpcb)
                expected=next(iter(evaluate(g,inputs).values())).reshape(-1)
                np.testing.assert_allclose(out.data[:count],expected,rtol=1e-12,atol=1e-12)
                tiled=out.data.reshape(-1,period)[:,:count]
                np.testing.assert_allclose(tiled,np.broadcast_to(expected,tiled.shape),rtol=1e-12,atol=1e-12)
                self.assertEqual(record["slot_count"],SLOTS)
                self.assertEqual(record["geometry"]["ni"],1)
                self.assertFalse(record["bootstrap_removed"])
            self.assertEqual(g,before)
    def test_unsupported_geometry_and_wrong_graph_rejected(self):
        with self.assertRaisesRegex(ValueError,"batch one"):bind(model((2,3)))
        b=Builder([(1,3)]);out=b.node("square",["input0"])
        with self.assertRaisesRegex(ValueError,"Standalone"):bind(b.finish(out))
        g=model((1,3));g["outputs"][0]["value"]="input0"
        with self.assertRaisesRegex(ValueError,"Standalone"):bind(g)
    def test_all_frozen_bn_models_have_exact_bindings(self):
        from benchmark_runner import load,DEFAULT
        rows,_=load(DEFAULT);bn=[r for r in rows if r["metadata"].get("helper")=="HE_BN"]
        self.assertEqual(len(bn),8)
        for row in bn:
            spec=bind(row["model"])
            self.assertEqual(spec["input_shape"],spec["output_shape"])
            self.assertEqual(spec["geometry"]["nt"],16384)

if __name__=="__main__":unittest.main()
