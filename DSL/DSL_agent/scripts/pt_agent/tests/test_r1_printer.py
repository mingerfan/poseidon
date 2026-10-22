from pathlib import Path
import unittest

from pt_agent.dsl.ir import canonical
from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program


EXAMPLES = Path(__file__).parents[1] / "examples"
EXPECTED_MLP = '''@fhe.program(slots=8192, refresh="compiler", placement="depth-dp")
def mlp(x: Cipher[1, 64]):
    v0 = pack(x, layout="flat", gap=1)
    v1 = linear(v0, w=W["fc1.weight"], b=W["fc1.bias"], algo="bsgs")
    v2 = act(v1, fn="square", method="exact")
    v3 = linear(v2, w=W["fc2.weight"], b=W["fc2.bias"], algo="diag")
    return v3
'''


class PrinterTests(unittest.TestCase):
    def test_exact_mlp_text(self):
        self.assertEqual(print_program(parse((EXAMPLES / "mlp.ptdsl").read_text())), EXPECTED_MLP)

    def test_stable_round_trips(self):
        for model in ("mlp", "resnet20"):
            with self.subTest(model=model):
                p = parse((EXAMPLES / f"{model}.ptdsl").read_text())
                first = print_program(p)
                second = print_program(parse(first))
                self.assertEqual(first, second)
                self.assertEqual(canonical(p), canonical(parse(first)))

    def test_expanded_region_and_defaults(self):
        text = print_program(parse((EXAMPLES / "resnet20.ptdsl").read_text()))
        self.assertIn('with region("stem", level_budget=16):', text)
        self.assertIn("domain=(-16.0, 16.0)", text)
        self.assertIn("bn_eps=1e-05", text)
        self.assertNotIn("def basic_block", text)

    def test_split_outputs(self):
        text = '''@fhe.program(slots=8192)
def split_sample(x: Cipher[1, 64]):
    a = pack(x, layout="flat")
    b, c = split(a, axis=1, sizes=(32, 32))
    d = add(b, c)
    return d
'''
        first = print_program(parse(text))
        self.assertIn("v1, v2 = split(v0, axis=1, sizes=(32, 32))", first)
        self.assertEqual(first, print_program(parse(first)))
        self.assertEqual(canonical(parse(text)), canonical(parse(first)))
