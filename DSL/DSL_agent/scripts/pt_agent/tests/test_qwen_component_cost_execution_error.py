"""A real backend failure aborts the retained transaction before quality/C2."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.backends.base import LevelError
from pt_agent.experiments import qwen_retained_component_replay as entry
from pt_agent.planners.transformer_session import load_state
from pt_agent.tests import test_qwen_retained_component_replay as fixture


class CostExecutionErrorChecks(unittest.TestCase):
    def test_backend_failure_keeps_authoritative_checkpoint_and_never_confirms(self):
        case=fixture.RetainedComponentChecks()
        case.setUp();self.addCleanup(case.doCleanups)
        case.prepared();seen=[];functions=case.original_functions(seen)
        original_evaluate=functions['evaluate'].__closure__[0].cell_contents
        # The fixture uses the original trusted evaluate AST. Make its caught
        # backend class the same actual LevelError ancestry used by production.
        original_evaluate.__globals__['BackendError']=entry.component.__dict__.get('BackendError',
            __import__('pt_agent.backends.base',fromlist=['BackendError']).BackendError)
        ctx=SimpleNamespace(prompt_version='r8-v11',placement='program-plan',profile='qwen48',
            weights={},weight_shapes={},graph=object(),model_info=object())
        error=LevelError(7,'fixture negative level')
        with case.numeric_seams(),patch.object(entry.guarded.bridge,'_agent_functions',return_value=functions), \
                patch.object(entry.component,'estimate',side_effect=error),patch.object(entry,'seed_partial'), \
                patch.object(entry.guarded.bridge,'_context',return_value=ctx):
            with self.assertRaises(entry.ComponentCostExecutionError)as caught:
                entry._agent_phase(case.args,case.recorder,{})
        self.assertIs(caught.exception.__cause__,error)
        self.assertEqual(load_state(case.session/'SESSION.json')['calls'],12)
        self.assertEqual(load_state(case.session/'SESSION.json')['phase'],'evaluate')
        self.assertNotIn('heldout',seen);self.assertNotIn('C2',seen)
        self.assertFalse((case.session/'component-c2-access.json').exists())
        self.assertFalse((case.session/'FROZEN.json').exists())

    def test_interruption_keeps_original_exception(self):
        original=InterruptedError('fixture stop')
        with patch.object(entry.guarded.bridge,'_agent_functions',return_value={}), \
                patch.object(entry.lifetime,'adapt_agent_functions',side_effect=lambda fn,**kw:kw['costs']), \
                patch.object(entry.component,'estimate',side_effect=original), \
                patch.object(entry.guarded.bounded,'FeedbackPresenter',return_value=None):
            # Feedback function is a required original namespace key.
            with patch.object(entry.lifetime,'adapt_agent_functions',
                    side_effect=lambda fn,**kw:dict(kw['costs'],feedback_message=object())):
                functions=entry.agent_functions('/tmp/component-error-fixture',object())
            with self.assertRaises(InterruptedError)as caught:functions['estimate'](object(),{})
        self.assertIs(caught.exception,original)


if __name__=='__main__':unittest.main()
