"""Deterministic small projection fixtures; no original models or data."""
from contextlib import ExitStack,nullcontext
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pt_agent import paths
from pt_agent.tests._artifacts import require


TEMPLATE=paths.RESULTS_DIR/'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json'


@require(TEMPLATE,paths.ORBIT_ROOT.parent/'orbit-venv/bin/python')
class QwenPrefillProjectionCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
        from pt_agent.experiments.r29_local_compile_matrix import native_metadata_commands
        cls.context=ExitStack();cls.addClassCleanup(cls.context.close)
        cls.context.enter_context(native_metadata_commands(json.loads(TEMPLATE.read_text())))
        import numpy as np
        from pt_agent.experiments import qwen_prefill_compile as entry
        from pt_agent.backends import CountBackend,RefreshBackend
        from pt_agent.backends.base import Backend
        from pt_agent.backends.cost_dag import CostDag
        from pt_agent.backends.streaming_mock import StreamingMockBackend
        from pt_agent.backends.streaming_emit import StreamingEmitBackend
        from pt_agent.kernels import transformer,streaming_linear
        from pt_agent.kernels.compact_linear import compact_transformer_projections
        from pt_agent.kernels.layout import layout_blocks,multi_flat_layout,unpack_sequence
        cls.np=np;cls.entry=entry;cls.Count=CountBackend;cls.Refresh=RefreshBackend;cls.Base=Backend
        cls.Dag=CostDag;cls.Mock=StreamingMockBackend;cls.Emit=StreamingEmitBackend
        cls.tr=transformer;cls.stream=streaming_linear;cls.compact=staticmethod(compact_transformer_projections)
        cls.blocks=staticmethod(layout_blocks);cls.layout=staticmethod(multi_flat_layout);cls.unpack=staticmethod(unpack_sequence)

    def fixture(self,inputs=3,outputs=20):
        np=self.np
        x=np.arange(inputs,dtype=np.float64)/7.-.25
        matrix=(np.arange(outputs*inputs,dtype=np.float64).reshape(outputs,inputs)%19.-9.)/32.
        bias=np.arange(outputs,dtype=np.float64)/1024.
        layout=self.layout(32,inputs);packed=[]
        for start,stop,_ in self.blocks(layout):
            block=np.zeros(32,dtype=np.float64);block[:stop-start]=x[start:stop];packed.append(block)
        return x,matrix,bias,layout,packed

    def run_projection(self,backend,fixture):
        _,matrix,bias,layout,packed=fixture
        if hasattr(backend,'bind'):backend.bind(packed)
        values=[backend.input(i)for i in range(len(packed))]
        with self.entry._projection_backend_scope(),self.compact():
            output,layout=self.tr.linear(backend,values,layout,matrix,bias)
        values=self.tr.values(output,layout);backend.output(values)
        return values,layout

    def counted(self,fixture):
        backend=self.Count(32,boot_out_level=5);self.run_projection(backend,fixture)
        stats=backend.stats()
        plan=dict(calls=stats['ops_total']+len(fixture[-1]),substitutions=[],outputs=[],dead=[],boots=0)
        return stats,plan

    def test_original_guard_rejection_and_supported_refresh_inner_types(self):
        fixture=self.fixture();counter=self.Count(32,boot_out_level=5)
        _,plan=self.counted(fixture)
        wrapper=self.Refresh(counter,policy='plan',plan=plan,input_level=8,boot_in_level=1,boot_out_level=5)
        with self.assertRaisesRegex(ValueError,'Streaming projections'):
            self.stream.streaming_projection(wrapper,[],fixture[3],fixture[1],fixture[2])
        self.assertTrue(self.entry._projection_backend_allowed(wrapper))
        invalid=self.Refresh(self.Base(32),policy='plan',plan=plan,input_level=8,boot_in_level=1,boot_out_level=5)
        self.assertFalse(self.entry._projection_backend_allowed(invalid))

    def test_tall_multiblock_and_short_expansion_keep_original_float64_math_and_counts(self):
        original_linear=self.tr.linear;original_projection=self.stream.streaming_projection
        for inputs,outputs in ((3,20),(3,7),(17,40)):
            with self.subTest(inputs=inputs,outputs=outputs):
                fixture=self.fixture(inputs,outputs);stats,plan=self.counted(fixture)
                counted=self.Count(32,boot_out_level=5)
                active=self.Refresh(counted,policy='plan',plan=plan,input_level=8,boot_in_level=1,boot_out_level=5)
                self.run_projection(active,fixture)
                self.assertEqual(counted.stats()['counts'],stats['counts']);self.assertEqual(active.stats()['boots'],0)
                mock=self.Mock(32);mock.bind(fixture[-1])
                active=self.Refresh(mock,policy='plan',plan=plan,input_level=8,boot_in_level=1,boot_out_level=5)
                values,layout=self.run_projection(active,fixture)
                got=self.np.concatenate([value.value.value[:block.size]
                    for value,(_,_,block)in zip(values,self.blocks(layout))])
                expected=fixture[1]@fixture[0]+fixture[2]
                self.assertEqual(got.shape,expected.shape)
                self.assertTrue(self.np.all(self.np.abs(got-expected)<=1e-12+1e-12*self.np.abs(expected)))
                self.assertEqual(mock._constant_objects,{})
        self.assertIs(self.tr.linear,original_linear);self.assertIs(self.stream.streaming_projection,original_projection)

    def test_cost_dag_recording_retains_wrapped_calls_and_no_constant_identity_arrays(self):
        fixture=self.fixture();dag=self.entry._ephemeral_cost_dag(self.Dag,self.np)(32)
        active=self.Refresh(dag,policy='record',input_level=8,boot_in_level=1,boot_out_level=5)
        self.run_projection(active,fixture);dag.finalize()
        self.assertGreater(active.call_count,1);self.assertGreater(len(active.calls),1)
        self.assertTrue(any(kind=='rot'for kind in dag.kind));self.assertTrue(any(kind=='mul_cp'for kind in dag.kind))
        self.assertEqual(dag._constant_objects,{})
        self.assertTrue(all(value.data is None for value in dag._constants.values()))
        self.assertEqual(len(dag._outputs),2)

    def test_streamed_emit_refresh_dispatch_matches_count_without_native_trace(self):
        fixture=self.fixture();stats,plan=self.counted(fixture)
        with TemporaryDirectory()as temporary:
            backend=self.Emit(32,directory=Path(temporary)/'emit',frontend=Path(temporary)/'unused-expr.py')
            active=self.Refresh(backend,policy='plan',plan=plan,input_level=8,boot_in_level=1,boot_out_level=5)
            try:
                self.run_projection(active,fixture)
                self.assertEqual(backend.stats()['counts'],stats['counts'])
                self.assertEqual(backend.stats()['distinct_consts'],stats['distinct_consts'])
                self.assertEqual(backend.stats()['identity_cache_entries'],0)
                self.assertEqual(backend.stats()['retained_constant_arrays'],0)
                self.assertTrue(backend.write().is_file())
            finally:
                if not backend.finished:backend.close()

    def test_original_schedule_resource_adapter_keeps_count_and_mock_math(self):
        from pt_agent.experiments.qwen_agent_resources import original_schedule_scope
        def run(backend,fixture,scope):
            _,matrix,bias,layout,packed=fixture
            if hasattr(backend,'bind'):backend.bind(packed)
            values=[backend.input(i)for i in range(len(packed))]
            with scope():output,layout=self.tr.linear(backend,values,layout,matrix,bias)
            values=self.tr.values(output,layout);backend.output(values)
            return values,layout
        original=self.tr.linear;projection=self.stream.streaming_projection
        for shape in ((3,20),(3,7),(17,40)):
            fixture=self.fixture(*shape);before=self.Count(32);after=self.Count(32)
            run(before,fixture,nullcontext);run(after,fixture,original_schedule_scope)
            self.assertEqual(before.stats(),after.stats())
            mock=self.Mock(32);values,layout=run(mock,fixture,original_schedule_scope)
            got=self.np.concatenate([value.value[:block.size]for value,(_,_,block)in zip(values,self.blocks(layout))])
            expected=fixture[1]@fixture[0]+fixture[2]
            self.assertTrue(self.np.all(self.np.abs(got-expected)<=1e-12+1e-12*self.np.abs(expected)))
            self.assertEqual(after._constant_objects,{})
        self.assertIs(self.tr.linear,original);self.assertIs(self.stream.streaming_projection,projection)

    def test_private_original_estimate_and_plan_use_same_tiny_program(self):
        from pt_agent.experiments.qwen_agent_resources import private_cost_namespace,original_schedule_scope
        from pt_agent.dsl.parser import parse
        from pt_agent import cost
        program=parse('@fhe.program(slots=1024,refresh="compiler",placement="depth-dp")\n'
            'def fixture(x: Cipher[1,3]):\n'
            '    p=pack(x,layout="flat")\n'
            '    y=linear(p,w=W["weight"],b=W["bias"],algo="bsgs")\n'
            '    return y\n')
        fixture=self.fixture();weights=dict(weight=fixture[1],bias=fixture[2])
        expected=cost.estimate_normalized_fast(program,weights,profile='qwen48')
        private=private_cost_namespace()
        with original_schedule_scope():got=private['estimate_normalized_fast'](program,weights,profile='qwen48')
        for key in ('counts','ops_total','distinct_consts','boots_pred','latency_pred_us','max_depth','placement_model'):
            self.assertEqual(got[key],expected[key])
        self.assertIsNot(private['CostDag'],cost.CostDag if hasattr(cost,'CostDag')else None)


if __name__=='__main__':unittest.main()
