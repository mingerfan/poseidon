"""No native runtime or model API needed for syntax/type/layout rejection tests."""
import unittest
from hecate_contract import validate_function


def program(body):
    return '@hc.func("c")\ndef golden(x):\n' + "\n".join("    " + line for line in body.splitlines()) + "\n"


class HecateContractTests(unittest.TestCase):
    def test_valid_affine_and_no_execution_claim(self):
        result = validate_function(program("product = x * weight\nreturn product + bias"),
                                   {"weight": [1, -2, .5, 3], "bias": .125})
        self.assertEqual(result["outputs"], [{"kind": "cipher", "slot_period": 4}])
        self.assertFalse(result["compilation_checked"])
        self.assertFalse(result["encrypted_correctness_checked"])
        self.assertFalse(result["safe_to_execute_untrusted"])

    def test_dot_product_fragment(self):
        body = "p = x * w\na = p + p.rotate(1)\nb = a + a.rotate(2)\nreturn b + bias"
        result = validate_function(program(body), {"w": [1, 2, 3, 4], "bias": [0.125]})
        self.assertEqual(result["rotation_steps"], [1, 2])
        # Never claim that structural checking proved the reduction semantics.
        self.assertEqual(result["outputs"][0]["slot_period"], 4)

    def test_cipher_product_and_multi_cipher_return(self):
        result = validate_function(program("y = x * x\nreturn [x, y]"), {}, expected_outputs=2)
        self.assertEqual(len(result["outputs"]), 2)

    def test_no_module_side_effects_or_extra_functions(self):
        for source in ("import os\n" + program("return x"), program("return x") + "print(1)\n",
                       "@evil()\n" + program("return x"), program("return x") * 2):
            with self.assertRaises(ValueError):
                validate_function(source, {})

    def test_call_and_attribute_allowlist(self):
        for expr in ('open("x")', 'x.__class__', 'getattr(x, "rotate")(1)', 'hc.bootstrap(x)',
                     'x.rotate(3)', 'x.rotate(-1)', 'x.rotate(True)', 'x.rotate(step=1)',
                     'x.rotate(x)', 'eval("1")', 'x.reshape(4)', 'x[0]', 'x / x', 'x ** 2', '-x', 'x - x'):
            with self.subTest(expr=expr), self.assertRaises(ValueError):
                validate_function(program(f"return {expr}"), {})

    def test_no_control_flow_mutation_or_augmented_assignment(self):
        for body in ('x += x\nreturn x', 'if x:\n    return x\nreturn x',
                     'for i in range(4):\n    y = x + x\nreturn x',
                     'x = x + x\nreturn x', 'weight = x\nreturn x',
                     'global y\nreturn x', 'y = [v for v in x]\nreturn x',
                     'y = x\ny = x + x\nreturn y'):
            with self.assertRaises(ValueError):
                validate_function(program(body), {"weight": [1, 2, 3, 4]})

    def test_no_plain_outputs_or_unchecked_broadcast(self):
        for constants in ({"w": []}, {"w": [1, 2]}, {"w": [[1, 2], [3, 4]]},
                          {"w": float("nan")}, {"w": True}, {"w": 1025}, {"x": 1}, {"w": 10**1000}):
            with self.assertRaises(ValueError):
                validate_function(program("return x"), constants)
        for expr in ("w", "w + x", "w * w", "unknown + x"):
            with self.assertRaises(ValueError):
                validate_function(program(f"return {expr}"), {"w": 1})

    def test_signature_and_result_count(self):
        for source in (program("return x").replace('"c"', '"p"'),
                       program("return x").replace("golden(x)", "golden(x=1)"),
                       program("return x").replace("golden(x)", "golden(x: int)"),
                       program("return x").replace("golden(x)", "golden(x, y)"),
                       program("return [x, x]"), program("y = x")):
            with self.assertRaises(ValueError):
                validate_function(source, {})

    def test_limits(self):
        with self.assertRaises(ValueError):
            validate_function(" " * 65537, {})
        with self.assertRaises(ValueError):
            validate_function(program("return " + " + ".join(["x"] * 100)), {})
