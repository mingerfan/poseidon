"""Registered frames and logical arithmetic against independent coordinate oracles."""
import copy
import json
import unittest
from pathlib import Path
import numpy as np
from unified_binding_coordinate_guidance import (
    VERSION, DIRECT, BSGS, FRAMES, NAMES, route_frames, broadcast_context, coordinate_hint)
from component_contract import prepare_task, request_options, reconstruct_request
from unified_graph_contract import validate_request
from application_diagnostics import attach

class BindingFramesTests(unittest.TestCase):
    def routes(self, period, bindings):
        return dict(layout=dict(input_slot_period=period), constant_origins=dict(
            capacity_tensor_packing=dict(bindings=bindings)))

    def test_three_direct_contexts_match_selected_tensor_coordinates(self):
        contexts = (
            ("suffix_gather", 16, 3, ((2, 5, 2.0), (9, 12, -.75))),
            ("pool_gather", 32, 7, ((1, 8, .25), (29, 4, 1.5))),
            ("tensor_route", 8, 1, ((0, 1, 1.), (6, 7, -2.))),
        )
        for role, period, step, pairs in contexts:
            with self.subTest(role=role):
                source = np.arange(period, dtype=float) ** 2 + .3
                vector = np.zeros(period); expected = np.zeros(period)
                for destination, origin, weight in pairs:
                    vector[destination] = weight
                    expected[destination] = weight * source[origin]
                request = self.routes(period, [dict(role=role, step=step,
                    baby_step=step, giant_step=0)])
                self.assertEqual(route_frames(request)[0]["mask_frame"], "destination")
                actual = vector * np.roll(source, -step)
                np.testing.assert_array_equal(actual, expected)
                self.assertFalse(np.array_equal(np.roll(source * vector, -step), expected))

    def test_three_bsgs_contexts_match_destination_coordinates(self):
        contexts = (
            ("suffix_gather", 16, 1, 4, ((0, 5, .5), (7, 12, 2.))),
            ("pool_stream_gather", 32, 3, 8, ((2, 13, .25), (27, 6, -1.))),
            ("tensor_route", 8, 1, 2, ((0, 3, 2.), (6, 1, -.5))),
        )
        for role, period, baby, giant, pairs in contexts:
            with self.subTest(role=role):
                source = np.arange(period, dtype=float) ** 3 + .1
                vector = np.zeros(period); expected = np.zeros(period)
                for destination, origin, weight in pairs:
                    vector[(destination + giant) % period] = weight
                    expected[destination] = weight * source[origin]
                request = self.routes(period, [dict(role=role, step=baby+giant,
                    baby_step=baby, giant_step=giant)])
                self.assertEqual(route_frames(request)[0]["mask_frame"], "baby")
                actual = np.roll(vector * np.roll(source, -baby), -giant)
                np.testing.assert_array_equal(actual, expected)
                self.assertFalse(np.array_equal(vector * np.roll(source, -baby-giant), expected))

    def model(self, op):
        return dict(format="poseidon-model-graph-v1", id="broadcast_arithmetic",
            inputs=[dict(name="x",shape=[2,5]),dict(name="y",shape=[2,1])],constants={},
            nodes=[dict(id="n",op=op,inputs=["x","y"],attrs={},outputs=["z"])],
            outputs=[dict(name="out",value="z")])

    def test_add_subtract_multiply_broadcasts_are_selected_and_keep_rows_separate(self):
        from benchmark_math import evaluate
        for op in ("add", "subtract", "multiply"):
            model=self.model(op); context=broadcast_context(dict(model=model))
            self.assertEqual(context["count"],1)
            self.assertEqual(context["examples"][0]["op"],op)
            x=np.arange(10,dtype=float).reshape(2,5); y=np.array([[2.],[11.]])
            expected=np.empty((2,5))
            for row in range(2):
                for column in range(5):
                    a,b=x[row,column],y[row,0]
                    expected[row,column]={"add":a+b,"subtract":a-b,"multiply":a*b}[op]
            np.testing.assert_array_equal(evaluate(model,dict(x=x,y=y))["out"],expected)

    def test_no_broadcast_rule_for_equal_shapes(self):
        model=self.model("add");model["inputs"][1]["shape"]=[2,5]
        self.assertIsNone(broadcast_context(dict(model=model)))

    def test_plain_direct_routes_and_mixed_frames(self):
        r=self.routes(16,[dict(role="suffix_gather",step=2),
            dict(role="suffix_gather",step=3,baby_step=3,giant_step=0),
            dict(role="suffix_gather",step=5,baby_step=1,giant_step=4)])
        self.assertEqual(route_frames(r),[
            dict(role="suffix_gather",mask_frame="baby",bindings=1),
            dict(role="suffix_gather",mask_frame="destination",bindings=2)])

    def test_incomplete_or_inconsistent_coordinates_rejected(self):
        for fields in (dict(step=2,baby_step=2),dict(step=2,baby_step=2,giant_step=1),
                       dict(step=True),dict(step=16),dict(step=2,baby_step=-1,giant_step=3)):
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    route_frames(self.routes(16,[dict(role="suffix_gather",**fields)]))

    def requests(self):
        p=Path(__file__).resolve().parent/"benchmarks/tools/linear_phase_live_v1/prepared/wide_tail_linear/request.json"
        old=json.loads(p.read_bytes());options=request_options(old)
        options["generation_guidance"]=VERSION
        return old,prepare_task(old["model"],options)

    def test_real_request_preserves_every_nonguidance_binding(self):
        old,new=self.requests();unchanged=copy.deepcopy(old)
        validate_request(old);validate_request(new)
        self.assertEqual(reconstruct_request(new),new)
        self.assertNotEqual(old["request_id"],new["request_id"])
        for key,value in old.items():
            if key not in ("generation_guidance","request_id"):
                self.assertEqual(new[key],value,key)
        self.assertEqual(old,unchanged)
        guidance=new["generation_guidance"]
        self.assertEqual(guidance["logical_arithmetic_broadcast"]["examples"][0]["op"],"add")
        self.assertEqual(guidance["bound_route_coordinate_frames"]["roles"],
            [dict(role="suffix_gather",mask_frame="destination",bindings=2)])
        self.assertIn("bound_tile_phase_frames",guidance)
        self.assertEqual(guidance["reserved_binding_rule"],NAMES)

    def test_feedback_separates_frames_without_private_data(self):
        _,request=self.requests()
        source=request["generation_guidance"]["entry_header"]+"\n    return x0, x1\n"
        candidate=dict(schema=1,request_id=request["request_id"],hecate_source=source)
        result=attach(dict(status="failed",failure=dict(layer="numerical_comparison",
            diagnostic="PRIVATE /tmp/reference")),candidate=candidate,request=request)
        text=result["public_feedback"]["diagnostic"]
        self.assertIn("destination-frame",text)
        self.assertIn("moving selection happens before",text)
        self.assertIn("exact-values-v1",text)
        self.assertNotIn("PRIVATE",text);self.assertNotIn("/tmp/reference",text)

    def test_reserved_names_feedback_is_fixed_and_public(self):
        _,request=self.requests()
        source=request["generation_guidance"]["entry_header"]+"\n    p1 = x0\n    return x0, x1\n"
        result=attach(dict(status="failed",failure=dict(layer="static_check",
            diagnostic="PRIVATE /tmp/reference")),candidate=dict(schema=1,
            request_id=request["request_id"],hecate_source=source),request=request)
        text=result["public_feedback"]["diagnostic"]
        self.assertIn("reserved bindings",text)
        self.assertNotIn("PRIVATE",text)

if __name__=="__main__":
    unittest.main(verbosity=2)
