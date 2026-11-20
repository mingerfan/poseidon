"""Real original Agent step bodies with deterministic metadata-only callbacks."""
import ast
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from pt_agent.experiments import qwen_agent_resources as entry
from pt_agent.experiments.qwen_bounded_feedback import FeedbackPresenter
from pt_agent.experiments.qwen_bridge_agent import _clone_namespace


class NamespaceDispatchChecks(unittest.TestCase):
    def fixture(self):
        # Execute unmodified AST function bodies from the actual trusted source.
        # Model/typecheck/cost/validation callbacks are deterministic metadata.
        source = Path(entry.__file__).parents[1]/'planners/agent.py'
        tree = ast.parse(source.read_text())
        names = ('agent_step', '_evaluate_step', '_finish_iteration')
        module = ModuleType('trusted_original_agent_fixture')
        namespace = module.__dict__
        namespace['__package__'] = 'pt_agent.planners'
        namespace.update(json=json, Path=Path, time=time, LLMError=RuntimeError,
            validate_tune_rule=lambda *a:None, apply_placement=lambda source, ctx:source,
            json_safe=lambda value:value, PRE_TUNE_RULE='v8-pre-6.11', feedback_message=lambda *a, **k:'original feedback')
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in selected}, set(names))
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
        tools = ModuleType('pt_agent.planners.agent_tools')
        exec('def evaluate(source,ctx,**kwargs): return estimate_normalized_fast(source,ctx)\n', tools.__dict__)
        def bridge(original_evaluate):
            def evaluate(source,ctx,**kwargs):
                ctx.append('candidate gate')
                return original_evaluate(source,ctx,**kwargs)
            return evaluate
        callback = bridge(tools.evaluate)
        namespace['evaluate'] = callback
        functions = _clone_namespace(module, {})
        state = dict(phase='evaluate', iteration=13, calls=12, max_iters=15, tuner=False, failed=None,
            engine_profile='tensor', tune_search='sweep', tune_rule='v8-6.25', prompt_version='r8-v11',
            prompt_tokens=0, completion_tokens=0, llm_min=None, best=None, best_program=None,
            has_tuned=False, history=[], messages=[], pool=[], pending=None, elapsed_s=0., stale=0, tune_policy='all')
        return functions, callback, state

    @contextmanager
    def tune_import(self):
        module = ModuleType('pt_agent.planners.transformer_tuner')
        module.validate_tune_search = lambda ctx:'sweep'
        with mock.patch.dict('sys.modules', {module.__name__:module}): yield

    def test_actual_agent_step_dispatches_private_cost_and_full_feedback_once(self):
        functions, original_callback, state = self.fixture()
        calls, active = [], []
        @contextmanager
        def scope():
            active.append(True)
            try: yield
            finally: active.pop()
        def cost(source,ctx):
            self.assertEqual(active, [True]); calls.append(source)
            return dict(stage='parse', feasible=False, exact=None, cost=None, validation=None, error='synthetic diagnostic')
        adapted = entry.adapt_agent_functions(functions,
            dict.fromkeys(('estimate','estimate_normalized','estimate_fast','estimate_normalized_fast'), cost), scope)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            adapted['feedback_message'] = FeedbackPresenter(directory/'full-feedback', adapted['feedback_message'])
            trace = directory/'trace.jsonl'; ctx = []
            with self.tune_import(): adapted['agent_step'](ctx, state, trace_path=trace,
                client=None, reply=dict(content=json.dumps(dict(program='metadata only')), usage={}, seconds=0))
            self.assertEqual(calls, ['metadata only']); self.assertEqual(ctx, ['candidate gate']); self.assertEqual(active, [])
            self.assertTrue((directory/'full-feedback/iteration-0013.full.json').is_file())
            self.assertEqual(state['calls'], 13); self.assertEqual(state['iteration'], 14)
            self.assertEqual(len(state['history']), 1); self.assertEqual(state['history'][0]['stage'], 'parse')
            self.assertEqual(len(trace.read_text().splitlines()), 1)
        for name in ('agent_step','_evaluate_step','_finish_iteration'):
            self.assertIs(adapted[name].__globals__, adapted)
            self.assertIs(adapted[name].__code__, functions[name].__code__)
            self.assertIs(functions[name].__globals__, functions)
        self.assertIs(functions['evaluate'], original_callback)
        self.assertNotIsInstance(functions['feedback_message'], FeedbackPresenter)

    def test_actual_step_exception_cleans_scope_and_preserves_original_namespace(self):
        functions, callback, state = self.fixture(); active = []
        @contextmanager
        def scope():
            active.append(True)
            try: yield
            finally: active.pop()
        def cost(*a): raise ValueError('deliberate metadata callback failure')
        adapted = entry.adapt_agent_functions(functions,
            dict.fromkeys(('estimate','estimate_normalized','estimate_fast','estimate_normalized_fast'), cost), scope)
        with tempfile.TemporaryDirectory() as directory, self.tune_import():
            with self.assertRaisesRegex(ValueError, 'deliberate'):
                adapted['agent_step']([], state, trace_path=Path(directory)/'trace', client=None,
                    reply=dict(content=json.dumps(dict(program='metadata only')), usage={}, seconds=0))
        self.assertEqual(active, []); self.assertIs(functions['evaluate'], callback)
        self.assertEqual(state['history'], [])


if __name__ == '__main__': unittest.main()
