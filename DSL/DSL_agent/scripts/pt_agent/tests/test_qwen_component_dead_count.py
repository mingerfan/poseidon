"""New tiny original DAG/plan/Count tests; no model, API or compiler."""
import gc
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from pt_agent import plan
from pt_agent.backends.base import BackendError, LevelError
from pt_agent.backends.count import CountBackend
from pt_agent.backends.refresh import RefreshBackend
from pt_agent.experiments import qwen_component_cost as entry


def dead_run(active):
    x = active.input(0)
    dead = x
    for _ in range(7):
        dead = dead * dead
    live = x + x
    active.output([live])
    return live


def original_plan(run=dead_run):
    dag, recorder, _ = plan.record_dag(run, 64, frontend_normalize=True)
    placement = plan._placement(dag, 'qwen48')
    return plan.express(dag, recorder, placement)


def physical(placement):
    inner = CountBackend(64, input_level=5, boot_out_level=5)
    return inner, RefreshBackend(inner, policy='plan', plan=placement, **plan.levels())


class DeadCountChecks(unittest.TestCase):
    def test_original_logical_verify_passes_but_physical_dead_chain_fails(self):
        placement = original_plan()
        self.assertEqual(placement['dead'], [[1, 8]])
        self.assertEqual(plan.verify(dead_run, 64, placement)['boots'], 0)
        inner, active = physical(placement)
        with self.assertRaises(LevelError):
            dead_run(active)
        self.assertEqual(inner.stats()['counts']['mul_cc'], 5)

    def test_dead_fix_retains_every_raw_call_and_live_physical_levels(self):
        placement = original_plan()
        inner, active = physical(placement)
        with entry._dead_count_scope(active):
            output = dead_run(active)
        self.assertEqual(inner.stats()['counts']['mul_cc'], 7)
        self.assertEqual(inner.stats()['counts']['add'], 1)
        self.assertEqual(inner.stats()['max_depth'], 7)
        self.assertEqual(inner.stats()['ops_total'], 8)
        self.assertEqual(active.call_count, placement['calls'])
        self.assertEqual(output.level, 5)
        self.assertEqual(output.value.level, 5)
        self.assertEqual(inner.input_level, 5)
        self.assertNotIn('_apply', inner.__dict__)
        self.assertNotIn('_component_dead_count_scope', inner.__dict__)

    def test_live_deep_chain_without_refresh_still_rejected(self):
        def run(active):
            value = active.input(0)
            for _ in range(7):value = value * value
            active.output([value])
        placement = original_plan(run)
        placement.update(substitutions=[], outputs=[], boots=0)
        inner, active = physical(placement)
        with self.assertRaisesRegex(BackendError, 'below boot_in_level'):
            with entry._dead_count_scope(active):run(active)
        self.assertEqual(inner.input_level, 5)
        self.assertNotIn('_apply', inner.__dict__)

    def test_planned_live_boots_keep_original_physical_level_checks(self):
        def run(active):
            value = active.input(0)
            for _ in range(7):value = value * value
            active.output([value])
        placement = original_plan(run)
        inner, active = physical(placement)
        with entry._dead_count_scope(active):run(active)
        self.assertGreater(placement['boots'], 0)
        self.assertEqual(inner.stats()['counts']['boot'], placement['boots'])
        self.assertEqual(inner._outputs[0].level, 2)

    def test_raw_live_count_failure_keeps_identity_and_safe_context(self):
        inner, active = physical(original_plan())
        raw = inner.input(0)
        raw.level = 0
        active._seq = 8  # Original live add's call index.
        with self.assertRaises(LevelError) as caught:
            with entry._dead_count_scope(active):inner._apply('mul_cc', raw, raw)
        self.assertEqual(caught.exception.component_call, dict(seq=8,plan_dead=False,
            kind='mul_cc',raw_counter_total=0,op_context=None,cipher_operand_levels=[0,0]))
        self.assertEqual(raw.level, 0)
        self.assertEqual(inner.input_level, 5)

    def test_dead_boot_exemption_and_post_scope_boundary_boot(self):
        inner, active = physical(original_plan())
        raw = inner.input(0);active._seq = 1
        with entry._dead_count_scope(active):
            value = raw.bootstrap()
            self.assertEqual(value.level, 5)
        self.assertEqual(value.bootstrap().level, 5)
        self.assertEqual(inner.stats()['counts']['boot'], 2)

    def test_async_exception_original_callable_once_and_finally_restore(self):
        inner, active = physical(original_plan())
        active._seq = 1
        raw = inner.input(0);calls=[];error=InterruptedError('tiny supervisor fixture')
        def original(kind,left,right=None,*,step=None):
            calls.append((kind, left, right, step, inner.input_level))
            raise error
        inner._apply = original
        with self.assertRaises(InterruptedError) as caught:
            with entry._dead_count_scope(active):inner._apply('mul_cc',raw,raw)
        self.assertIs(caught.exception,error)
        self.assertEqual(len(calls),1)
        self.assertIs(calls[0][1],raw)
        self.assertIsNone(calls[0][4])
        self.assertIs(inner._apply,original)
        self.assertEqual(inner.input_level,5)

    def test_original_return_identity_and_runtime_signature(self):
        inner, active = physical(original_plan());raw=inner.input(0);token=object();calls=[]
        def original(kind,left,right=None,*,step=None):calls.append((kind,step));return token
        inner._apply=original
        with entry._dead_count_scope(active):
            self.assertEqual(str(inspect.signature(inner._apply)), '(kind, left, right=None, *, step=None)')
            self.assertIs(inner._apply('rot',raw,step=3),token)
        self.assertEqual(calls,[('rot',3)])
        self.assertIs(inner._apply,original)

    def test_nested_and_invalid_dead_indices_rejected_without_hook_mutation(self):
        inner, active = physical(original_plan())
        with entry._dead_count_scope(active):
            installed=inner._apply
            with self.assertRaises(RuntimeError):
                with entry._dead_count_scope(active):pass
            self.assertIs(inner._apply,installed)
        for bad in (-1,active._plan.calls,True,'1'):
            active._plan.dead={bad}
            with self.assertRaises(ValueError):
                with entry._dead_count_scope(active):pass
            self.assertNotIn('_apply',inner.__dict__)

    def test_callback_dispatch_preserves_source_weights_and_lower_return(self):
        from pt_agent.experiments import qwen_layered_structure as layered
        segment=SimpleNamespace(program=object(),weight_shapes=())
        weights={};callback=entry.already_scoped_lowering(layered.SegmentLowering(segment,weights))
        inner, active=physical(original_plan());token=object()
        def lower(program, actual_weights, actual_active, *, act_mode):
            self.assertIs(program,segment.program);self.assertIs(actual_weights,weights)
            self.assertIs(actual_active,active);self.assertEqual(act_mode,'program')
            self.assertIn('_apply',inner.__dict__)
            return token
        with patch.object(layered,'source_binding',return_value={}) as source, \
                patch('pt_agent.lower.lower',side_effect=lower):
            self.assertIs(callback(active),token)
        source.assert_called_once_with(segment)
        self.assertNotIn('_apply',inner.__dict__)

    def test_disabled_gc_no_installed_wrapper_retains_backend(self):
        enabled=gc.isenabled();gc.disable()
        try:
            inner, active=physical(original_plan());refs=weakref.ref(inner),weakref.ref(active)
            with entry._dead_count_scope(active):dead_run(active)
            from pt_agent.experiments.qwen_agent_cost_lifetime import release_backend
            release_backend(inner)  # Original owner-table lifecycle, unchanged.
            del inner,active
            self.assertTrue(all(ref() is None for ref in refs))
        finally:
            if enabled:gc.enable()

    def test_failed_replay_file_is_bounded_and_never_success_artifacts(self):
        error=LevelError(0,'fixture')
        error.component_call=dict(seq=8,plan_dead=False,kind='mul_cc',raw_counter_total=0,
            op_context=None,cipher_operand_levels=[0,0])
        with tempfile.TemporaryDirectory() as temp:
            entry._write_count_error(temp,'layer0',error)
            self.assertEqual(sorted(p.name for p in Path(temp).iterdir()),['layer0-count-error.json'])
            saved=json.loads((Path(temp)/'layer0-count-error.json').read_text())
            self.assertEqual(saved,dict(status='error',planned_verified=True,complete_count=False,
                error='LevelError',component_call=error.component_call))
        stop=InterruptedError('write supervisor fixture')
        with patch('pt_agent.experiments.r10_paper_metrics.write_json',side_effect=stop):
            with self.assertRaises(InterruptedError) as caught:entry._write_count_error('/tmp','layer0',error)
        self.assertIs(caught.exception,stop)
        with patch('pt_agent.experiments.r10_paper_metrics.write_json',side_effect=OSError('optional disk fixture')):
            entry._write_count_error('/tmp','layer0',error)


if __name__=='__main__':unittest.main()
