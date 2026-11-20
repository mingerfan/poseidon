"""Original tiny callback plans and physical Count; no model or compiler."""
import copy
import gc
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'

import numpy as np
from pt_agent.backends.base import BackendError,LevelError
from pt_agent.backends.refresh import RefreshBackend
from pt_agent.backends.streaming_count import StreamingCountBackend
from pt_agent.experiments import qwen_generation_kernels as entry


# Existing command-construction schema only; no compiler or identifier check.
METADATA=['{hecate_opt}','{input}','--pars','--waterline=43',
    '--ckks-config={package}/config/compiler-config.json',
    '--runtime-plan-context-id=qwen-bootstrap48',
    '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
    '--runtime-plan-operator-spec-sha256=sha256:'+'0'*64,
    '-o','{output_dir}/lowered.mlir']


class TinyKernel:
    kind='terminal'
    geometry=dict(slots=64,vocabulary_blocks=1)
    input_count=1

    def __init__(self,live_deep=False):self.live_deep=live_deep
    def source(self):return dict(kind=self.kind,own_tiny_fixture=True,live_deep=self.live_deep)
    def output_roles(self):return ['own_tiny_live']
    def __call__(self,active):
        x=active.input(0);deep=x
        for _ in range(7):deep=deep*deep
        active.output([deep if self.live_deep else x+x])


def planned(kernel):
    return entry.plan_kernel(kernel,metadata_template=METADATA)[0]


class KernelDeadCountChecks(unittest.TestCase):
    def test_original_logical_verify_and_unadapted_physical_dead_error(self):
        kernel=TinyKernel();bound=planned(kernel)
        self.assertEqual(bound.refresh_plan['dead'],[[1,8]])
        self.assertEqual(bound.replay['boots'],0)
        physical=StreamingCountBackend(64,input_level=5,boot_out_level=5)
        active=RefreshBackend(physical,policy='plan',plan=bound.refresh_plan,**entry.LEVELS)
        with self.assertRaises(LevelError):kernel(active)
        self.assertEqual(physical.stats()['counts']['mul_cc'],5)

    def test_run_kernel_retains_all_dead_raw_calls_and_original_live_level(self):
        kernel=TinyKernel();bound=planned(kernel);before=copy.deepcopy(bound)
        physical=StreamingCountBackend(64,input_level=5,boot_out_level=5)
        output,interface=entry.run_kernel(physical,kernel,bound)
        stats=physical.stats()
        self.assertEqual(stats['counts']['mul_cc'],7);self.assertEqual(stats['counts']['add'],1)
        self.assertEqual(stats['ops_total'],8);self.assertEqual(stats['max_depth'],7)
        self.assertEqual(interface['refresh']['calls'],bound.refresh_plan['calls'])
        self.assertEqual(interface['refresh']['boots'],0)
        self.assertEqual(interface['symbolic_output_levels'],[5]);self.assertEqual(output[0].level,5)
        self.assertEqual(physical.input_level,5)
        self.assertNotIn('_apply',physical.__dict__)
        self.assertNotIn('_component_dead_count_scope',physical.__dict__)
        self.assertEqual(bound,before);self.assertEqual(interface['source'],kernel.source())

    def test_live_deep_missing_original_refresh_is_still_rejected(self):
        kernel=TinyKernel(live_deep=True);bound=planned(kernel)
        self.assertGreater(bound.refresh_plan['boots'],0)
        invalid=copy.deepcopy(bound.refresh_plan)
        invalid.update(substitutions=[],outputs=[],boots=0)
        bad=entry.PlannedKernel(kernel.source(),invalid,
            dict(calls=invalid['calls'],boots=0,inner_boots=0))
        physical=StreamingCountBackend(64,input_level=5,boot_out_level=5)
        with self.assertRaisesRegex(BackendError,'below boot_in_level'):
            entry.run_kernel(physical,kernel,bad)
        self.assertEqual(physical.input_level,5);self.assertNotIn('_apply',physical.__dict__)

    def test_original_live_refreshes_keep_count_and_output_level_checks(self):
        kernel=TinyKernel(live_deep=True);bound=planned(kernel)
        physical=StreamingCountBackend(64,input_level=5,boot_out_level=5)
        output,interface=entry.run_kernel(physical,kernel,bound)
        self.assertGreater(bound.refresh_plan['boots'],0)
        self.assertEqual(physical.stats()['counts']['boot'],bound.refresh_plan['boots'])
        self.assertEqual(physical.stats()['counts']['mul_cc'],7)
        self.assertEqual(output[0].level,interface['symbolic_output_levels'][0])
        self.assertTrue(1<=output[0].level<=5)

    def test_async_callback_preserves_error_and_restores_the_physical_count_hook(self):
        kernel=TinyKernel();bound=planned(kernel);error=InterruptedError('own tiny cancellation')
        class InterruptedKernel(TinyKernel):
            def __call__(self,active):
                super().__call__(active)
                raise error
        physical=StreamingCountBackend(64,input_level=5,boot_out_level=5)
        with self.assertRaises(InterruptedError)as caught:
            entry.run_kernel(physical,InterruptedKernel(),bound)
        self.assertIs(caught.exception,error);self.assertEqual(physical.input_level,5)
        self.assertEqual(physical.stats()['counts']['mul_cc'],7)
        self.assertNotIn('_apply',physical.__dict__)
        self.assertNotIn('_component_dead_count_scope',physical.__dict__)

    def test_original_first_and_terminal_tiny_geometry_plan_and_real_Count(self):
        geometry=entry._geometry(8,5,1,3)
        state=dict(step=0,prefix_length=1,prefix_token_ids=[0,0,0],sample_id='own-tiny:0',
            attention_mask=[[1.,0.,0.]],sel=[[1.,0.,0.]])
        binding=entry.SourceBinding(dict(initial_state=state,geometry={k:geometry[k]
            for k in ('slots','vocabulary','hidden','capacity')},own_tiny_fixture=True))
        matrix=np.asarray([[.125],[-.25],[.375],[-.5],[.625]],dtype=np.float64);matrix.flags.writeable=False
        first=entry.Kernel('first',binding,matrix,geometry)
        terminal=entry.Kernel('terminal',binding,None,geometry)
        try:
            for kernel in (first,terminal):
                with self.subTest(kind=kernel.kind):
                    source=kernel.source();bound=planned(kernel)
                    counted=entry.count_kernel(kernel,bound,metadata_template=METADATA)
                    self.assertEqual(kernel.source(),source)
                    config=source['selector']
                    self.assertEqual(config['stage_count'],52);self.assertEqual(config['degrees'],[3]*52)
                    self.assertEqual(config['difference_bound'],32.)
                    self.assertEqual(config['normalized_gap'],2.**-27)
                    self.assertEqual(counted['stats']['counts']['boot'],bound.refresh_plan['boots'])
                    self.assertEqual(counted['interface']['refresh']['calls'],bound.refresh_plan['calls'])
                    self.assertEqual(counted['interface']['inputs'],3 if kernel.kind=='first'else 2)
                    self.assertEqual(counted['interface']['outputs'],5 if kernel.kind=='first'else 2)
                    self.assertTrue(all(1<=level<=5 for level in counted['interface']['symbolic_output_levels']))
                    self.assertEqual(counted['interface']['explicit_boundary_bootstraps_in_this_adapter'],0)
                    self.assertIs(first._matrix,matrix);self.assertFalse(matrix.flags.writeable)
        finally:
            first.close();terminal.close();gc.collect()

    def test_original_streaming_emit_is_unpatched_and_keeps_all_raw_calls(self):
        from pt_agent.backends.streaming_emit import StreamingEmitBackend
        kernel=TinyKernel();bound=planned(kernel)
        counted=entry.count_kernel(kernel,bound,metadata_template=METADATA)
        with tempfile.TemporaryDirectory()as directory:
            physical=StreamingEmitBackend(64,directory=Path(directory)/'emit',
                frontend=Path(directory)/'unused-own-frontend.py',max_constant_bytes=4096)
            try:
                _,interface=entry.run_kernel(physical,kernel,bound)
                self.assertNotIn('_apply',physical.__dict__)
                self.assertNotIn('_component_dead_count_scope',physical.__dict__)
                stats=physical.stats()
                self.assertEqual(stats['counts'],counted['stats']['counts'])
                self.assertEqual(stats['ops_total'],8)
                self.assertEqual(stats['counts']['mul_cc'],7)
                self.assertEqual(interface,counted['interface'])
                self.assertTrue(physical.write().is_file())
            finally:
                if not physical.finished:physical.close()


if __name__=='__main__':unittest.main()
