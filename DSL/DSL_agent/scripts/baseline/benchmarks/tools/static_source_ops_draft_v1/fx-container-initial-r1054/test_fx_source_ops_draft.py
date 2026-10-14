"""Known-fixture FX intake checks for the isolated Agent draft."""
import operator
import unittest

import torch
from torch.nn import functional as F

from fx_source_ops_draft import from_fx_call
from source_ops_draft import SourcePlanError, verify_call
from test_source_ops_draft import APPROX


class Fixture(torch.nn.Module):
    def __init__(self, call, divisor=2., pool=None):
        super().__init__()
        self.call, self.divisor, self.pool = call, divisor, pool
        if type(divisor) is list:
            self.register_buffer("public_divisor", torch.tensor(divisor, dtype=torch.float64))

    def forward(self, x):
        if self.pool is not None:
            rank, kernel, stride, padding, dilation = self.pool
            fn = F.max_pool1d if rank == 1 else F.max_pool2d
            return fn(x, kernel, stride, padding, dilation, False, False)
        divisor = self.public_divisor if type(self.divisor) is list else self.divisor
        if self.call == "operator.truediv":
            return operator.truediv(x, divisor)
        if self.call.startswith("Tensor."):
            return getattr(x, self.call.split(".")[1])(divisor)
        return getattr(torch, self.call.split(".")[1])(x, divisor)


def capture(fixture, shape, approximation=None):
    graph = torch.fx.symbolic_trace(fixture.eval())
    operands, calls = {}, []
    for node in graph.graph.nodes:
        if node.op == "placeholder":
            operands[node] = dict(kind="cipher", name="x", shape=shape, dtype="float64")
        elif node.op == "get_attr":
            # This fixture only registers a public divisor. No arbitrary paths,
            # methods or state dictionary execution are admitted by the helper.
            if node.target != "public_divisor":
                raise ValueError("Unexpected fixture constant")
            public = graph.get_buffer("public_divisor")
            if public.dtype is not torch.float64:
                raise ValueError("Unexpected fixture precision")
            operands[node] = dict(kind="public", value=public.tolist())
        elif node.op in ("call_function", "call_method"):
            calls.append(node)
        elif node.op != "output":
            raise ValueError("Unexpected fixture node")
    if len(calls) != 1:
        raise ValueError("Fixture must contain exactly one tested call")
    call = from_fx_call(calls[0], operands, approximation=approximation)
    verify_call(call, expected_source_call_sha256=call["source_call_sha256"])
    return graph, call


class FxSourceDraftTests(unittest.TestCase):
    def assert_rejection(self, code, node, bindings):
        with self.assertRaises(SourcePlanError) as cm:
            from_fx_call(node, bindings, approximation=APPROX)
        self.assertEqual(cm.exception.code, code)
        self.assertNotIn("private_source", str(cm.exception.feedback()))

    def test_actual_division_fx_aliases(self):
        for alias in ("operator.truediv", "torch.div", "torch.divide", "torch.true_divide",
                      "Tensor.div", "Tensor.divide", "Tensor.true_divide"):
            with self.subTest(alias=alias):
                graph, call = capture(Fixture(alias), [2, 3])
                self.assertEqual(call["source_call"]["call"], alias)
                self.assertEqual(call["plan"]["source_request"]["divisor"], 2.)
                self.assertFalse(call["plan"]["production_intake_integrated"])

    def test_actual_boolean_dispatch_pooling_targets(self):
        for rank in (1, 2):
            with self.subTest(rank=rank):
                graph, call = capture(Fixture(None, pool=(rank, 2, 1, 0, 1)),
                                      [1, 1] + [4]*rank, APPROX)
                self.assertEqual(call["source_call"]["call"], "F.max_pool"+str(rank)+"d")

    def test_public_tensor_buffer(self):
        graph, call = capture(Fixture("Tensor.divide", [[2., -4., 8.]]), [2, 3])
        self.assertEqual(call["plan"]["source_request"]["divisor"], [[2., -4., 8.]])

    def test_function_name_spoof_rejected(self):
        def private_source(x, k):
            return x
        private_source.__name__ = "max_pool2d"
        private_source.__module__ = "torch.nn.functional"
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_function(private_source, (x, 2))
        self.assert_rejection("source_call_unregistered", node,
                              {x: dict(kind="cipher", name="x", shape=[1,1,4,4], dtype="float64")})

    def test_unbound_operand_rejected(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_function(torch.div, (x, 2.))
        self.assert_rejection("source_fx_operand_unbound", node, {})

    def test_encrypted_denominator_stays_rejected(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_function(torch.div, (x, x))
        self.assert_rejection("division_cipher_denominator_unsupported", node,
                              {x: dict(kind="cipher", name="x", shape=[2], dtype="float64")})

    def test_unknown_method_stays_rejected(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_method("exp", (x,))
        self.assert_rejection("source_call_unregistered", node,
                              {x: dict(kind="cipher", name="x", shape=[2], dtype="float64")})

    def test_metadata_object_hooks_are_not_called(self):
        class Unsafe:
            def __deepcopy__(self, memo):
                raise AssertionError("Untrusted object hook executed")
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_function(torch.div, (x, 2.))
        self.assert_rejection("source_call_arguments_rejected", node,
                              {x: dict(kind="cipher",name="x",shape=[2],dtype=Unsafe())})

    def test_cyclic_metadata_is_rejected(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_function(torch.div, (x, 2.))
        metadata = dict(kind="cipher",name="x",shape=[2],dtype="float64")
        metadata["cycle"] = metadata
        self.assert_rejection("source_call_arguments_rejected", node, {x: metadata})

    def test_method_other_keyword(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_method("div", (x,), {"other": 2., "rounding_mode": None})
        call = from_fx_call(node, {x: dict(kind="cipher",name="x",shape=[2],dtype="float64")})
        self.assertEqual(call["plan"]["source_request"]["divisor"], 2.)


if __name__ == "__main__":
    unittest.main()
