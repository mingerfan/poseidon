from collections import Counter
from pathlib import Path
import unittest

from pt_agent.dsl.errors import CHECK_CODES, PARSE_CODES, PTDSLError
from pt_agent.dsl.ir import Header, Param, WeightRef
from pt_agent.dsl.ops_spec import OPS
from pt_agent.dsl.parser import parse
from pt_agent.dsl.typecheck import check
from pt_agent.examples.shapes import mlp_weight_shapes


EXAMPLES = Path(__file__).parents[1] / "examples"


class ParserTests(unittest.TestCase):
    def test_mlp_records(self):
        p = parse((EXAMPLES / "mlp.ptdsl").read_text())
        self.assertEqual([op.kind for op in p.ops], ["pack", "linear", "act", "linear"])
        self.assertEqual(p.header, Header(slots=8192, refresh="compiler", placement="depth-dp",
                                          l_max=None, boot_out_level=None, lineno=2))
        self.assertEqual(p.params, (Param("x", "cipher", (1, 64), 3),))
        self.assertEqual(len(p.outputs), 1)

    def test_resnet20_expansion(self):
        p = parse((EXAMPLES / "resnet20.ptdsl").read_text())
        self.assertEqual(len(p.ops), 57)
        self.assertEqual(Counter(op.kind for op in p.ops), dict(conv2d=19, act=19, add=9,
            slice=4, pad=2, pack=1, mean=1, relayout=1, linear=1))
        self.assertEqual(list(p.regions), ["stem"])
        self.assertEqual(p.regions["stem"].level_budget, 16)
        self.assertEqual(sum(op.region == "stem" for op in p.ops), 2)
        conv = next(op for op in p.ops if op.kind == "conv2d" and op.macro_stack == ("basic_block",))
        self.assertEqual(conv.attrs["w"], WeightRef("layer1.0.conv1.weight"))
        self.assertEqual(conv.attrs["bn"], "layer1.0.bn1")
        self.assertEqual(conv.attrs["stride"], (1, 1))
        self.assertEqual(conv.attrs["dilation"], (1, 1))
        self.assertEqual(conv.attrs["groups"], 1)
        self.assertEqual(conv.attrs["bn_eps"], 1e-5)
        self.assertEqual([op.attrs["pads"] for op in p.ops if op.kind == "pad"],
                         [(0, 0, 0, 0, 8, 8), (0, 0, 0, 0, 16, 16)])
        act = next(op for op in p.ops if op.kind == "act")
        self.assertEqual(act.attrs["domain"], (-16.0, 16.0))
        self.assertEqual(act.attrs["degrees"], (15, 15, 27))

    def test_ssa_and_producers(self):
        p = parse((EXAMPLES / "mlp.ptdsl").read_text())
        self.assertEqual(p.ops[3].inputs, p.ops[2].outputs)
        values = [vid for op in p.ops for vid in op.outputs]
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual(p.producer(p.ops[2].outputs[0]), p.ops[2])
        self.assertIsNone(p.producer("x"))
        self.assertEqual(p.param("x"), p.params[0])

    def test_invalid_parse_fixtures(self):
        files = sorted((EXAMPLES / "invalid").glob("*.ptdsl"))
        self.assertEqual(len(files), 29)
        for path in files:
            text = path.read_text()
            if not text.startswith("# expect:"):
                continue
            expected = text.splitlines()[0].split(":", 1)[1].strip()
            lineno = next((i for i, line in enumerate(text.splitlines(), 1) if "# ERR" in line), 0)
            with self.subTest(file=path.name):
                with self.assertRaises(PTDSLError) as caught:
                    parse(text)
                self.assertEqual(caught.exception.code, expected)
                self.assertEqual(caught.exception.lineno, lineno)

    def test_diagnostic_code_sets(self):
        self.assertEqual(len(OPS), 27)
        for path in sorted((EXAMPLES / "invalid").glob("*.ptdsl")):
            with self.subTest(file=path.name):
                try:
                    p = parse(path.read_text())
                except PTDSLError as exc:
                    self.assertIn(exc.code, PARSE_CODES)
                else:
                    for diag in check(p, mlp_weight_shapes()).diags:
                        self.assertIn(diag.code, CHECK_CODES)

    def test_no_dynamic_source_execution(self):
        for path in (Path(__file__).parents[1] / "dsl").glob("*.py"):
            for forbidden in ("exec(", "eval(", "compile("):
                with self.subTest(file=path.name, forbidden=forbidden):
                    self.assertNotIn(forbidden, path.read_text())
        main = '@fhe.program(slots=8192)\ndef entry(x: Cipher[1, 64]):\n    return x\n'
        for body, code in (
            ('    for i in range(2):\n        a = act(a, fn="square", method="exact")\n    return a\n', "E_STMT"),
            ('    with region("hidden", level_budget=1):\n        b = act(a, fn="square", method="exact")\n    return b\n', "E_REGION"),
        ):
            with self.subTest(macro=code), self.assertRaises(PTDSLError) as caught:
                parse(main + '\ndef unused(a):\n' + body)
            self.assertEqual(caught.exception.code, code)
        text = ('@fhe.program(slots=8192)\ndef entry(x: Cipher[1, 64]):\n'
                '    y = f(x, W[missing])\n    return y\n\ndef f(a, p):\n    return a\n')
        with self.assertRaises(PTDSLError) as caught:
            parse(text)
        self.assertEqual(caught.exception.code, "E_MACRO")
