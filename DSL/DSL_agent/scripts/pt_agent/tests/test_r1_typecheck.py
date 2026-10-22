from pathlib import Path
import unittest

from pt_agent.dsl.parser import parse
from pt_agent.dsl.typecheck import check
from pt_agent.examples.shapes import mlp_weight_shapes, resnet20_weight_shapes


EXAMPLES = Path(__file__).parents[1] / "examples"


class TypecheckTests(unittest.TestCase):
    def test_mlp(self):
        p = parse((EXAMPLES / "mlp.ptdsl").read_text())
        result = check(p, mlp_weight_shapes())
        self.assertTrue(result.ok, result.diags)
        self.assertEqual(result.shapes[p.outputs[0]], (1, 10))
        self.assertFalse(any(d.code == "W_UNUSED" for d in result.diags))

    def test_resnet20(self):
        p = parse((EXAMPLES / "resnet20.ptdsl").read_text())
        weights = resnet20_weight_shapes()
        self.assertEqual(len(weights), 97)
        result = check(p, weights)
        self.assertTrue(result.ok, result.diags)
        self.assertEqual(result.shapes[p.outputs[0]], (1, 10))
        conv = next(op for op in p.ops if op.kind == "conv2d" and op.attrs["stride"] == (2, 2))
        pad = next(op for op in p.ops if op.kind == "pad")
        mean = next(op for op in p.ops if op.kind == "mean")
        self.assertEqual(result.shapes[conv.outputs[0]], (1, 32, 16, 16))
        self.assertEqual(result.shapes[pad.outputs[0]], (1, 32, 16, 16))
        self.assertEqual(result.shapes[mean.outputs[0]], (1, 64))

    def test_invalid_check_fixtures(self):
        for path in sorted((EXAMPLES / "invalid").glob("*.ptdsl")):
            text = path.read_text()
            if not text.startswith("# expect-check:"):
                continue
            code = text.splitlines()[0].split(":", 1)[1].strip()
            lineno = next(i for i, line in enumerate(text.splitlines(), 1) if "# ERR" in line)
            with self.subTest(file=path.name):
                result = check(parse(text), mlp_weight_shapes())
                self.assertTrue(any(d.code == code and d.lineno == lineno and d.severity == "error"
                                    for d in result.diags), result.diags)

    def test_changed_weight_shape_line(self):
        p = parse((EXAMPLES / "mlp.ptdsl").read_text())
        weights = mlp_weight_shapes()
        weights["fc1.weight"] = (32, 63)
        result = check(p, weights)
        self.assertTrue(any(d.code == "E_WEIGHT_SHAPE" and d.lineno == 5 for d in result.diags))

    def test_unused_value_warning(self):
        text = '''@fhe.program(slots=8192)
def unused(x: Cipher[1, 64]):
    a = pack(x, layout="flat")
    b = act(a, fn="square", method="exact")
    return a
'''
        result = check(parse(text), {})
        self.assertTrue(result.ok)
        self.assertTrue(any(d.code == "W_UNUSED" and d.lineno == 4 for d in result.diags))

    def test_compiler_refresh_warning(self):
        text = '''@fhe.program(slots=8192)
def refreshed(x: Cipher[1, 64]):
    a = pack(x, layout="flat")
    b = refresh(a)
    return b
'''
        result = check(parse(text), {})
        self.assertTrue(result.ok)
        self.assertTrue(any(d.code == "W_REFRESH_COMPILER" for d in result.diags))

    def test_no_cascaded_shape_errors(self):
        text = (EXAMPLES / "invalid/tc_shape.ptdsl").read_text()
        text = text.replace("    return c", '    d = act(c, fn="square", method="exact")\n    return d')
        result = check(parse(text), mlp_weight_shapes())
        self.assertEqual(sum(d.code == "E_SHAPE" for d in result.diags), 1)
