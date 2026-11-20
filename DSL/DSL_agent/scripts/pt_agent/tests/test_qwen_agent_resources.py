"""Finite stdlib dependency/closure checks; no model or numerical execution."""
from contextlib import contextmanager
from types import ModuleType
import unittest

from pt_agent.experiments import qwen_agent_resources as entry


def _cost_fixture(mode):
    if mode=='plan':
        from .plan import plan_program,verify_program
        return plan_program(),verify_program()
    from .backends.cost_dag import CostDag
    return CostDag


class QwenAgentResourceChecks(unittest.TestCase):
    def test_private_cost_dependencies_and_wrapper_estimators_leave_original_globals(self):
        module=ModuleType('trusted_cost_fixture');module.estimate=_cost_fixture
        exec('def estimate_normalized(value): return estimate(value)\n'
             'def estimate_fast(value): return estimate(value)\n'
             'def estimate_normalized_fast(value): return estimate_normalized(value)\n',module.__dict__)
        dag=type('PrivateDag',(),{});count=type('PrivateCount',(),{})
        planner=dict(CostDag=dag,CountBackend=count,plan_program=lambda:'original plan',verify_program=lambda:'original verify')
        original=dict(vars(module));private=entry.private_cost_namespace(module,planner)
        self.assertEqual(private['estimate_normalized_fast']('plan'),('original plan','original verify'))
        self.assertIs(private['estimate']('dag'),dag);self.assertIs(private['CountBackend'],count)
        self.assertEqual(vars(module),original)

    def test_dependency_drift_fails_before_any_evaluator_action(self):
        with self.assertRaisesRegex(ValueError,'dependencies changed'):
            entry._estimate_delegate(self.test_dependency_drift_fails_before_any_evaluator_action,{})

    def test_private_bridge_retains_gate_and_scope_on_success_and_failure(self):
        module=ModuleType('pt_agent.planners.agent_tools')
        exec('def evaluate(source,ctx,**kwargs):\n    return estimate_normalized_fast(source,ctx)\n',module.__dict__)
        def bridge(original_evaluate):
            def evaluate(source,ctx,**kwargs):
                ctx.append('candidate gate')
                return original_evaluate(source,ctx,**kwargs)
            return evaluate
        callback=bridge(module.evaluate);active=[]
        @contextmanager
        def scope():
            active.append(True)
            try:yield
            finally:active.pop()
        def cost(source,ctx):
            self.assertEqual(active,[True]);ctx.append('unchanged cost')
            if source=='failure':raise RuntimeError('fixture failure')
            return 'original outcome'
        costs=dict.fromkeys(('estimate','estimate_normalized','estimate_fast','estimate_normalized_fast'),cost)
        functions=dict(evaluate=callback,confirm_select=object())
        adapted=entry.adapt_agent_functions(functions,costs,scope)
        original=module.evaluate;trace=[]
        self.assertEqual(adapted['evaluate']('success',trace),'original outcome')
        self.assertEqual(trace,['candidate gate','unchanged cost']);self.assertEqual(active,[])
        with self.assertRaisesRegex(RuntimeError,'fixture failure'):adapted['evaluate']('failure',[])
        self.assertEqual(active,[]);self.assertIs(module.evaluate,original)
        self.assertIs(functions['evaluate'],callback);self.assertIs(adapted['confirm_select'],functions['confirm_select'])

    def test_closure_drift_is_explicit_and_original_function_unchanged(self):
        def unrelated(source,ctx):return source
        with self.assertRaisesRegex(ValueError,'original private bridge'):
            entry.adapt_agent_functions(dict(evaluate=unrelated),{})
        with self.assertRaisesRegex(ValueError,'closure changed'):entry._replace_closure(unrelated,'original_evaluate',object())
        self.assertEqual(unrelated('original',None),'original')


if __name__=='__main__':unittest.main()
