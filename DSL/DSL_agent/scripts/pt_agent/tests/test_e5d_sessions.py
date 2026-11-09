"""E5d bounded real-session wiring, exercised only with local clients."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.cost import estimate_fast
from pt_agent.dsl.parser import parse
from pt_agent.experiments import a1_transformer_agent as entry
from pt_agent.llm import LLMError
from pt_agent.paths import RESULTS_DIR
from pt_agent.planners.agent_tools import apply_placement
from pt_agent.planners.prompts import build_system_prompt
from pt_agent.planners.transformer_session import ModelBudgetClient, load_state, model_call_offset, output_tokens
from pt_agent.tests._artifacts import require
from pt_agent.tests.test_r8c_feedback import SOURCE, context


class SessionWiringTests(unittest.TestCase):
    def test_provider_tokens_offset_and_model_limits(self):
        for version in ('r8-v8', 'r8-v9', 'r8-v10', 'r8-v11'):
            self.assertEqual(output_tokens(version), 65536)
        for model in ('bert', 'qwen'):
            self.assertEqual(model_call_offset(model, 'r8-v11'), 0)
        self.assertEqual(entry.session_iterations('r8-v11', 0), 10)
        self.assertEqual(entry.session_iterations('r8-v11', 9), 1)
        self.assertEqual(entry.session_iterations('r8-v10', 0), 8)
        self.assertEqual(entry.call_limit('r8-v10'), 40)
        with self.assertRaises(ValueError):entry.session_iterations('r8-v11', 10)
        with TemporaryDirectory() as directory:
            stub=SimpleNamespace(chat=lambda *a,**k:dict(content='{}'))
            client=ModelBudgetClient(stub, directory, model='bert', prompt_version='r8-v11')
            for _ in range(10):client.chat([],purpose='offline-test')
            with self.assertRaises(LLMError):client.chat([],purpose='offline-test')
            self.assertEqual(client.calls,10)
            self.assertEqual(len(list((Path(directory)/'bert/r8-v11').glob('*.request.json'))),10)

    def test_missing_or_incompatible_placement_sends_zero_requests(self):
        with patch('pt_agent.llm.LLMClient') as provider:
            for version,placement in (('r8-v11', None), ('r8-v10', 'program-plan')):
                args=SimpleNamespace(prompt_version=version,placement=placement)
                with self.assertRaises(ValueError):entry.split_session(args)
                with self.assertRaises(ValueError):entry.session(args)
            provider.assert_not_called()

    def test_prompt_replacement_rejects_missing_and_duplicate_originals(self):
        from pt_agent.planners import prompts
        ctx=context();ctx.prompt_version='r8-v11'
        old='bootstraps, computed with the same bootstrap placement as the compiler, and'
        for replacement in ('removed',old+old):
            with patch.object(prompts,'TEMPLATE',prompts.TEMPLATE.replace(old,replacement)):
                with self.assertRaises(ValueError):build_system_prompt(ctx)

    def test_exhausted_toy_refresh_reports_zero_unused_levels(self):
        ctx=context();ctx.placement='program-plan'
        # Four consumed multiplications reach boot_in_level=1 exactly.
        source='''@fhe.program(slots=16384, refresh="agent", placement="program-plan", l_max=5, boot_out_level=5)
def toy(v: Cipher[1, 2]):
    x = pack(v, layout="flat")
    a = mul(x, x)
    b = mul(a, a)
    c = mul(b, b)
    d = mul(c, c)
    e = mul(d, d)
    f = mul(e, e)
    return f
'''
        cost=estimate_fast(parse(source),ctx.weights)
        values=[row['mean'] for row in cost['refresh_plan']['levels_left_at_refresh']]
        self.assertIn(0,values)
        self.assertTrue(all(value>=0 for value in values))


MODEL=RESULTS_DIR/'A1-bert-prepare-v2-a620-20261030/model'
TRACE=RESULTS_DIR/'A1-bert-session-20261029T055416-4780/agent_trace.jsonl'


@require(MODEL/'model_info.json', MODEL/'calibration.json', TRACE)
class FirstStepTests(unittest.TestCase):
    def test_split_session_first_evaluate_uses_v11_prompt_and_namespace(self):
        from pt_agent.models.info import ModelInfo
        from pt_agent.planners.agent_tools import build_context
        row=json.loads(TRACE.read_text().splitlines()[0])
        self.assertEqual(row['model_call'],35)
        reply=dict(content=row['messages'][0]['content'],usage=row['usage'],seconds=0.,finish_reason='stop')
        seen=[]
        class StubClient:
            def chat(self,messages,*,purpose):
                seen.append(copy.deepcopy(messages))
                return copy.deepcopy(reply)
        class FirstStepComplete(Exception):pass
        with TemporaryDirectory() as directory:
            root=Path(directory);budget=root/'calls.jsonl';budget.write_text('')
            args=SimpleNamespace(model='bert',model_dir=MODEL,prompt_version='r8-v11',placement='program-plan',
                out=root/'session',root=None,call_dir=root/'calls',budget_path=budget,
                tune_rule='v8-6.24',resume_session=None,response_file=None)
            def run_step(command,**kwargs):
                if command[command.index('--step')+1]!='evaluate':raise FirstStepComplete()
                self.assertEqual(command[command.index('--placement')+1],'program-plan')
                step_dir=Path(command[command.index('--out')+1])
                inner=SimpleNamespace(**vars(args));inner.root=args.out;inner.out=step_dir;inner.step='evaluate'
                result=entry.session_step(inner)
                entry._write(step_dir/'REPORT.json',result)
                return SimpleNamespace(wait=lambda **kw:0)
            with patch('pt_agent.llm.LLMClient',return_value=StubClient()) as provider, \
                    patch.object(entry.subprocess,'Popen',side_effect=run_step), \
                    patch.object(entry,'account_calls'):
                with self.assertRaises(FirstStepComplete):entry.split_session(args)
            state=load_state(args.out/'SESSION.json')
            self.assertEqual((state['max_iters'],state['placement'],state['prompt_version']),
                             (10,'program-plan','r8-v11'))
            self.assertEqual(len(seen),1)
            info=ModelInfo.load(MODEL)
            ctx=build_context(info,entry.read(MODEL/'calibration.json'),samples=entry.samples_for(MODEL,info))
            ctx.prompt_version='r8-v11';ctx.placement='program-plan'
            self.assertEqual(seen[0][0]['content'],build_system_prompt(ctx))
            provider.assert_called_once_with(budget,model='deepseek-flash',reasoning_effort='high',
                                            max_tokens=65536,max_total_calls=450)
            request=args.call_dir/'bert/r8-v11/call-001.request.json'
            self.assertTrue(request.exists())
            self.assertEqual(state['phase'],'tune')
            feedback=state['pending']['fb']
            self.assertEqual(feedback['stage'],'complete')
            self.assertIn('placement_variant',feedback['cost'])
            self.assertTrue(all('levels_left_at_refresh' in h for h in feedback['cost']['hotspots']))
            self.assertTrue((args.out/'sources/iteration-1.ptdsl').exists())


if __name__=='__main__':unittest.main()
