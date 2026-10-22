import json
from pathlib import Path
import unittest

from pt_agent.dsl.ir import canonical
from pt_agent.dsl.parser import parse
from pt_agent.dsl.typecheck import check
from pt_agent.intake import Graph
from pt_agent.planners.agent_tools import build_context, evaluate
from pt_agent.planners.prompts import PROMPT_VERSION, build_system_prompt
from pt_agent.planners.rule import plan
from pt_agent.paths import pt_result


R3 = pt_result('r3-resnet20-20261020T131658')
TRACE = pt_result('run-resnet20-agent-20261021T031807/agent_trace.jsonl')


class FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = Graph.load(R3/'graph')
        cls.calibration = json.loads((R3/'calibration.json').read_text())
        cls.ctx = build_context(cls.graph, cls.calibration, n_val=50)
        row = json.loads(TRACE.read_text().splitlines()[0])
        content = next(message['content'] for message in row['messages'] if message['role'] == 'assistant')
        cls.source = json.loads(content)['program']

    def test_original_feedback_locates_macro_call_and_shapes(self):
        feedback = evaluate(self.source, self.ctx)
        self.assertEqual(feedback['stage'], 'typecheck')
        error = feedback['errors'][0]
        self.assertEqual(error['code'], 'E_SHAPE')
        self.assertTrue(error['call_lines'])
        self.assertIn('down(h, "layer2.0"', error['call_texts'][0])
        self.assertEqual(error['line_text'], 't = add(t, s)')
        self.assertIn('(1, 32, 16, 16)', error['message'])
        self.assertIn('(1, 48, 16, 16)', error['message'])
        diag = next(diag for diag in check(parse(self.source), self.ctx.weight_shapes).diags
                    if diag.severity == 'error')
        self.assertIn('line 37 (in macro down called at line 9)', str(diag))

    def test_corrected_channel_padding_passes_typecheck(self):
        corrected = self.source.replace('down(h, "layer2.0", 16,', 'down(h, "layer2.0", 8,')
        corrected = corrected.replace('down(h, "layer3.0", 32,', 'down(h, "layer3.0", 16,')
        self.assertNotEqual(evaluate(corrected, self.ctx)['stage'], 'typecheck')

    def test_prompt_version_padding_and_example(self):
        self.assertEqual(PROMPT_VERSION, 'r8-v4')
        prompt = build_system_prompt(self.ctx)
        self.assertIn('C -> C + 2p', prompt)
        source = prompt.split('# example: residual_small\n', 1)[1].split('\n# example:', 1)[0]
        self.assertEqual(parse(source).name, 'tinynet')

    def test_nested_macro_call_stack_and_canonical(self):
        source = '''@fhe.program(slots=8192)
def entry(x: Cipher[1, 4], y: Cipher[1, 5]):
    a = pack(x, layout="flat")
    b = pack(y, layout="flat")
    c = outer(a, b)
    return c

def outer(a, b):
    c = inner(a, b)
    return c

def inner(a, b):
    c = add(a, b)
    return c
'''
        program = parse(source)
        self.assertEqual(program.ops[-1].call_lines, (5, 9))
        self.assertEqual(program.ops[-1].macro_stack, ('outer', 'inner'))
        diag = next(diag for diag in check(program, {}).diags if diag.severity == 'error')
        self.assertEqual(diag.call_lines, (5, 9))
        self.assertIn('line 13 (in macro inner called at line 9)', str(diag))
        self.assertEqual(canonical(program), canonical(parse('\n'+source)))

    def test_lowering_error_has_source_and_preserves_string(self):
        source = plan(self.graph, self.calibration, deg=127, margin=1.5)
        source = source.replace('stride=(1, 1)', 'stride=(2, 2)', 1)
        feedback = evaluate(source, self.ctx)
        self.assertEqual(feedback['stage'], 'lower')
        self.assertIn('line', feedback['error'])
        error = feedback['errors'][0]
        self.assertEqual(error['code'], 'E_LOWER')
        self.assertEqual(error['line_text'], source.splitlines()[error['line']-1].strip())
        self.assertEqual(error['call_lines'], [])
        self.assertEqual(error['call_texts'], [])
        self.assertIn(error['message'], feedback['error'])
