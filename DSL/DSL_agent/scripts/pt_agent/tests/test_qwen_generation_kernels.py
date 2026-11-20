"""Only synthetic protocol fixtures and tiny deterministic original arithmetic."""
from contextlib import contextmanager, nullcontext
import copy
import gc
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
from pt_agent.experiments import qwen_generation_kernels as entry


def initial(vocabulary=33, capacity=32, length=9):
    return dict(step=0, prefix_length=length, prefix_token_ids=[i % vocabulary for i in range(capacity)],
        sample_id='synthetic:0', attention_mask=[[1.]*length+[0.]*(capacity-length)],
        sel=[[0.]*(length-1)+[1.]+[0.]*(capacity-length)])


def binding(geometry=None):
    geometry = geometry or entry._geometry(64, 33, 2, 32)
    return entry.SourceBinding(dict(initial_state=initial(geometry['vocabulary'], geometry['capacity']),
        geometry={key:geometry[key] for key in ('slots','vocabulary','hidden','capacity')},
        frozen=dict(candidate_frozen=True, source_program_ast_verified=True), fixture=True))


class Value:
    __array_priority__ = 1000
    def __init__(self, backend, data):
        self.backend, self.data = backend, np.asarray(data, dtype=np.float64)
    def _other(self, other):
        if isinstance(other, Value):
            if other.backend is not self.backend:raise ValueError('foreign backend')
            return other.data
        return other
    def __add__(self, other):return Value(self.backend, self.data+self._other(other))
    __radd__ = __add__
    def __sub__(self, other):return Value(self.backend, self.data-self._other(other))
    def __rsub__(self, other):return Value(self.backend, self._other(other)-self.data)
    def __mul__(self, other):return Value(self.backend, self.data*self._other(other))
    __rmul__ = __mul__
    def __neg__(self):return Value(self.backend, -self.data)
    def rotate(self, k):return Value(self.backend, np.roll(self.data, -k))


class Numeric:
    """Small independent slot arithmetic; no production Count/Emit instances."""
    def __init__(self, slots, rows):
        self.slots, self.rows, self.indices, self._outputs = slots, rows, [], None
    def input(self, index):
        self.indices.append(index)
        return Value(self, self.rows[index])
    def const(self, value):return np.asarray(value, dtype=np.float64)
    def output(self, values):self._outputs = list(values)


@contextmanager
def numeric_scope():
    from pt_agent.kernels import compact_linear
    from pt_agent.experiments.qwen_prefill_compile import _projection_delegate
    original = compact_linear.embedding_from_selection
    delegated = _projection_delegate(original, lambda backend:isinstance(backend, Numeric))
    with patch.object(compact_linear, 'embedding_from_selection', delegated):yield


def plan(calls=7, boots=1):
    return dict(format='pt-agent-refresh-plan-v1', levels=dict(entry.LEVELS),
        calls=calls, boots=boots, outputs=[[0,41]] if boots else [], substitutions=[], dead=[])


class DummyKernel:
    def __init__(self, kind='first'):
        self.kind, self.geometry, self.input_count = kind, dict(slots=64, vocabulary_blocks=2), 3
    def source(self):return dict(kind=self.kind, stable='synthetic-source')
    def output_roles(self):return ['one_hot:0','one_hot:1','mixed_embedding','attention_mask','sel']
    def __call__(self, active):
        inputs = [active.input(i) for i in range(self.input_count)]
        active.output([active.make(i) for i in range(5)])


class Physical:
    instances = []
    def __init__(self, slots, **kwargs):
        self.slots, self._inputs, self._outputs = slots, [], None
        self._constants, self._constant_objects = {}, {}
        self.finished, self.closed, self.wrote, self.kwargs = False, False, False, kwargs
        self._stats = dict(counts=dict(boot=1, add=3), ops_total=4, distinct_consts=2)
        Physical.instances.append(self)
    def stats(self):return copy.deepcopy(self._stats)
    def write(self):
        self.wrote = self.finished = True
        return Path(self.kwargs['directory'])/'candidate_trace.py'
    def close(self):self.closed=True


class FakeRefresh:
    last = None
    def __init__(self, inner, **kwargs):
        self.inner, self.slots, self.shared = inner, inner.slots, kwargs['plan']
        self._plan = SimpleNamespace(outputs=dict(self.shared['outputs']))
        self._planned_cache, self._cache = {}, {}
        FakeRefresh.last = weakref.ref(self)
    def input(self, index):
        if index != len(self.inner._inputs):raise ValueError('bad input order')
        value = SimpleNamespace(index=index)
        self.inner._inputs.append(value)
        return SimpleNamespace(backend=self, value=value, level=5)
    def make(self, index):
        return SimpleNamespace(backend=self, value=SimpleNamespace(payload=index), level=2+index%3)
    def output(self, values):
        actual = list(values)
        for position,node in self._plan.outputs.items():
            selected = SimpleNamespace(backend=self, value=SimpleNamespace(payload=100+position), level=5)
            self._planned_cache[node] = selected
            actual[position] = selected
        self.inner._outputs = [value.value for value in actual]
    def stats(self):return dict(calls=self.shared['calls'], boots=self.shared['boots'])


def planned(kernel, **kwargs):
    p = plan(**kwargs)
    return entry.PlannedKernel(kernel.source(), p,
        dict(calls=p['calls'],boots=p['boots'],inner_boots=p['boots']))


def api(planner=None):
    return dict(Refresh=FakeRefresh, Count=Physical, Emit=Physical, planner=planner)


class SourceAndArithmeticChecks(unittest.TestCase):
    def test_state_exact_prefix_and_rejected_geometry(self):
        self.assertEqual(entry._state(initial())['prefix_length'],9)
        for key,value in [('step',False),('prefix_length',True),('prefix_length',31),
                          ('sel',[[0.]*32]),('attention_mask',[[1.]*32])]:
            state=initial();state[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):entry._state(state)
        bad=initial();bad['prefix_token_ids'][1]=-1
        with self.assertRaises(ValueError):entry._state(bad)
        for args in [(True,33,2,32),(63,33,2,32),(64,33,33,32),(16,33,2,32)]:
            with self.assertRaises(ValueError):entry._geometry(*args)

    def test_metadata_binding_uses_original_gate_and_same_prompt(self):
        frozen=dict(candidate_frozen=True,source_program_ast_verified=True,full_model_layers=24,sequence_capacity=32)
        with TemporaryDirectory() as tmp:
            case=Path(tmp);generation=dict(index=7,partition='calibration',sample_id='synthetic:0',prompt_length=9,sequence_length=32)
            (case/'generation-input.json').write_text(json.dumps(generation))
            with patch('pt_agent.experiments.qwen_prefill_compile.frozen_source',return_value=frozen) as gate:
                bound=entry.bind_source(case,initial())
                gate.assert_called_once_with(case.resolve())
                self.assertFalse(bound.snapshot()['input_payloads_verified'])
                generation['prompt_length']=10;(case/'generation-input.json').write_text(json.dumps(generation))
                with self.assertRaises(ValueError):entry.bind_source(case,initial())
                generation['prompt_length']=9;(case/'generation-input.json').write_text(json.dumps(generation))
                frozen['candidate_frozen']=False
                with self.assertRaises(ValueError):entry.bind_source(case,initial())

    def test_production_abi_19_blocks_tail4480_and_factory_geometry(self):
        geometry=entry._geometry(entry.SLOTS,entry.VOCABULARY,entry.HIDDEN,entry.CAPACITY)
        self.assertEqual(geometry['vocabulary_blocks'],19)
        self.assertEqual(geometry['valid_vocabulary_sizes'],[8192]*18+[4480])
        # Readonly broadcast is a zero-storage shape fixture, never numerical model input.
        matrix=np.broadcast_to(np.float64(0.),(entry.VOCABULARY,entry.HIDDEN))
        original={entry.WEIGHT:matrix,**{f'key:{i}':None for i in range(558)}}
        bound=binding(geometry)
        first,terminal=entry.make_kernels(bound,original)
        self.assertEqual((first.input_count,len(first.output_roles())),(20,22))
        self.assertEqual((terminal.input_count,len(terminal.output_roles())),(19,19))
        self.assertIs(first._matrix,matrix);self.assertIsNone(terminal._matrix)
        self.assertEqual(first.source()['future_boundary_bootstraps'],3)
        original.pop('key:0')
        with self.assertRaises(ValueError):entry.make_kernels(bound,original)
        first.close();terminal.close()

    def test_real_selector_compact_embedding_and_public_masks_tiny(self):
        from pt_agent.kernels.encrypted_generation import numpy_selection
        from pt_agent.experiments.qwen_monotone_selector import analytic_coefficients
        geometry=entry._geometry(64,33,2,32)
        matrix=(np.arange(66,dtype=np.float64).reshape(33,2)%11-5.)/16.
        matrix.flags.writeable=False
        kernel=entry.Kernel('first',binding(geometry),matrix,geometry)
        logits=(np.arange(33,dtype=np.float64)-16.)/8.
        rows=[np.pad(logits[:32],(0,32)),np.pad(logits[32:],(0,63)),np.arange(64,dtype=np.float64)/16.]
        physical=Numeric(64,rows)
        from pt_agent.kernels import compact_linear
        original=compact_linear.embedding_from_selection
        with patch.object(entry,'_kernel_scope',numeric_scope):kernel(physical)
        self.assertIs(compact_linear.embedding_from_selection,original)
        expected,_=numpy_selection(logits,bound=32.,normalized_gap=2.**-27,coefficients=analytic_coefficients(np))
        observed=np.concatenate([value.data[:n] for value,n in zip(physical._outputs[:2],[32,1])])
        np.testing.assert_allclose(observed,expected,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(physical._outputs[2].data[:2],expected@matrix,rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(physical._outputs[3].data,np.r_[np.ones(10),np.zeros(54)])
        np.testing.assert_array_equal(physical._outputs[4].data,np.r_[np.zeros(9),1.,np.zeros(54)])
        self.assertEqual(physical.indices,[0,1,2])
        self.assertEqual(int(observed.argmax()),32)

    def test_terminal_does_not_compute_or_hold_unused_embedding(self):
        geometry=entry._geometry(64,33,2,32)
        terminal=entry.Kernel('terminal',binding(geometry),None,geometry)
        logits=(np.arange(33,dtype=np.float64)-16.)/8.
        physical=Numeric(64,[np.pad(logits[:32],(0,32)),np.pad(logits[32:],(0,63))])
        with patch.object(entry,'_kernel_scope',lambda:nullcontext()), \
                patch('pt_agent.kernels.compact_linear.embedding_from_selection',side_effect=AssertionError('unused embedding')):
            terminal(physical)
        self.assertEqual(len(physical._outputs),2);self.assertEqual(physical.indices,[0,1])
        self.assertIsNone(terminal._matrix)
        with self.assertRaises(ValueError):entry.Kernel('terminal',binding(geometry),np.zeros((33,2)),geometry)

    def test_readonly_ownership_close_and_invalid_matrix(self):
        geometry=entry._geometry(64,33,2,32);matrix=np.zeros((33,2),dtype=np.float64)
        with self.assertRaises(ValueError):entry.Kernel('first',binding(geometry),matrix,geometry)
        matrix.flags.writeable=False;ref=weakref.ref(matrix)
        kernel=entry.Kernel('first',binding(geometry),matrix,geometry);del matrix
        self.assertIsNotNone(ref());kernel.close();self.assertIsNone(ref())
        with self.assertRaises(ValueError):kernel.source()
        with self.assertRaises(ValueError):kernel(Numeric(64,[]))


class ProtocolAndCleanupChecks(unittest.TestCase):
    def setUp(self):Physical.instances.clear()
    def contexts(self):
        return patch.object(entry,'_metadata',lambda template:nullcontext()),patch.object(entry,'_kernel_scope',lambda:nullcontext())

    def test_plan_record_express_verify_exact_order_and_clean_private_record(self):
        trace=[];refs={};shared=plan()
        class Dag:
            def __init__(self):
                self.numbering={1:'meta'};self._constant_objects={};self._constants={};self._inputs=[]
                self._outputs=[]
        def record(callback,slots,**kwargs):
            trace.append(('record',slots,kwargs));dag=Dag()
            recorder=SimpleNamespace(inner=dag,calls=[],output_nodes=[],_planned_cache={},_cache={},_plan=None)
            recorder._planned_cache['cycle']=recorder
            refs.update(dag=weakref.ref(dag))
            callback(recorder)
            return dag,recorder,.01
        class Callback:
            geometry=dict(slots=64)
            def source(self):return dict(source='bound')
            def __call__(self,recorder):trace.append(('callback',recorder.inner is not None))
        planner=dict(record_dag=record,_placement=lambda dag,profile:trace.append(('placement',profile)) or {},
            express=lambda *args,**kw:trace.append(('express',kw)) or shared,
            _merge=lambda *args:trace.append(('merge',)),
            verify=lambda *args,**kw:trace.append(('verify',refs['dag']() is None)) or dict(calls=7,boots=1,inner_boots=1))
        context1,context2=self.contexts()
        with context1,context2,patch.object(entry,'_api',return_value=api(planner)):
            result,metadata=entry.plan_kernel(Callback(),metadata_template=['fixed'])
        self.assertEqual([item[0] for item in trace],['record','callback','placement','express','merge','verify'])
        self.assertEqual(trace[-1],('verify',True))
        self.assertEqual(result.refresh_plan,shared);self.assertIsNot(result.refresh_plan,shared)
        self.assertEqual(metadata['fresh_model_loads'],0)

    def test_plan_callback_owns_one_dispatch_scope_without_reentrant_clone(self):
        depth=0;visits=[];shared=plan()
        @contextmanager
        def dispatch():
            nonlocal depth
            self.assertEqual(depth,0);depth+=1;visits.append('enter')
            try:yield
            finally:depth-=1;visits.append('exit')
        class Callback:
            geometry=dict(slots=64)
            def source(self):return dict(source='bound')
            def __call__(self,active):
                with entry._kernel_scope():pass
        def record(callback,*args,**kwargs):
            dag=SimpleNamespace();recorder=SimpleNamespace(inner=dag)
            callback(recorder)
            return dag,recorder,.01
        def verify(callback,*args,**kwargs):
            callback(SimpleNamespace(inner=SimpleNamespace()))
            return dict(calls=7,boots=1,inner_boots=1)
        planner=dict(record_dag=record,_placement=lambda *a:None,express=lambda *a,**k:shared,
            _merge=lambda *a:None,verify=verify)
        with patch.object(entry,'_metadata',lambda template:nullcontext()), \
                patch.object(entry,'_kernel_scope',dispatch),patch.object(entry,'_api',return_value=api(planner)):
            entry.plan_kernel(Callback(),metadata_template=[])
        self.assertEqual(visits,['enter','exit','enter','exit']);self.assertEqual(depth,0)

    def test_independent_verify_releases_private_wrapper_without_gc(self):
        refs={};shared=plan()
        class Callback:
            geometry=dict(slots=64)
            def source(self):return dict(source='bound')
            def __call__(self,active):pass
        def record(callback,*args,**kwargs):
            dag=SimpleNamespace();active=SimpleNamespace(inner=dag)
            callback(active)
            return dag,active,.01
        def verify(callback,*args,**kwargs):
            inner=Physical(64);active=FakeRefresh(inner,plan=shared)
            active._planned_cache[0]=active.make(0)
            callback(active)
            refs['active']=weakref.ref(active)
            return dict(calls=7,boots=1,inner_boots=1)
        planner=dict(record_dag=record,_placement=lambda *a:None,express=lambda *a,**k:shared,
            _merge=lambda *a:None,verify=verify)
        was=gc.isenabled();gc.disable()
        try:
            with patch.object(entry,'_metadata',lambda template:nullcontext()), \
                    patch.object(entry,'_api',return_value=api(planner)):
                entry.plan_kernel(Callback(),metadata_template=[])
            self.assertIsNone(refs['active']())
            self.assertEqual(Physical.instances[-1]._inputs,[])
            self.assertIsNone(Physical.instances[-1]._outputs)
        finally:
            if was:gc.enable()

    def test_record_callback_failure_still_clears_owned_cycles(self):
        owned={}
        class Dag:pass
        class Recorder:pass
        def record(callback,*args,**kwargs):
            dag=Dag();dag.numbering={};dag._constant_objects={};dag._constants={};dag._inputs=[]
            recorder=Recorder();recorder.inner=dag;recorder.calls=[];recorder._plan=None
            recorder._planned_cache={'cycle':recorder};recorder._cache={}
            owned['dag']=weakref.ref(dag);owned['recorder']=weakref.ref(recorder)
            callback(recorder)
        class Broken:
            geometry=dict(slots=64)
            def __call__(self,active):raise RuntimeError('callback failed')
        was=gc.isenabled();gc.disable()
        try:
            context1,context2=self.contexts()
            with context1,context2,patch.object(entry,'_api',return_value=api(dict(record_dag=record))):
                with self.assertRaisesRegex(RuntimeError,'callback failed'):
                    entry.plan_kernel(Broken(),metadata_template=[])
            self.assertIsNone(owned['dag']());self.assertIsNone(owned['recorder']())
        finally:
            if was:gc.enable()

    def test_effective_output_substitution_level_and_no_wrapper_escape(self):
        kernel=DummyKernel();selected=planned(kernel);original=copy.deepcopy(selected.refresh_plan)
        physical=Physical(64);was=gc.isenabled();gc.disable()
        try:
            raw,info=entry.run_kernel(physical,kernel,selected,refresh_class=FakeRefresh)
            self.assertEqual(info['symbolic_output_levels'],[5,3,4,2,3])
            self.assertEqual(raw[0].payload,100)
            self.assertEqual(info['caller_required_boundary_positions'],[2,3,4])
            self.assertEqual(info['original_token0_input'],2)
            self.assertFalse(info['native_ABI_verified'])
            self.assertIsNone(FakeRefresh.last())
            self.assertEqual(selected.refresh_plan,original)
            self.assertEqual(raw[4].payload,4)
        finally:
            if was:gc.enable()

    def test_bad_plan_source_counts_and_dirty_backend_rejected(self):
        kernel=DummyKernel();selected=planned(kernel)
        for modified in [entry.PlannedKernel(dict(wrong=True),selected.refresh_plan,selected.replay),
                         entry.PlannedKernel(selected.source,selected.refresh_plan,dict(calls=8,boots=1,inner_boots=1))]:
            with self.assertRaises(ValueError):entry.run_kernel(Physical(64),kernel,modified,refresh_class=FakeRefresh)
        physical=Physical(64);physical._inputs.append('already-used')
        with self.assertRaises(ValueError):entry.run_kernel(physical,kernel,selected,refresh_class=FakeRefresh)

    def test_callback_error_breaks_private_refresh_cycles_without_gc(self):
        class Broken(DummyKernel):
            def __call__(self,active):
                active.input(0)
                active._planned_cache[8]=active.make(8)
                active._cache['self']=active
                raise RuntimeError('synthetic kernel error')
        kernel=Broken();selected=planned(kernel);physical=Physical(64)
        was=gc.isenabled();gc.disable()
        try:
            with self.assertRaisesRegex(RuntimeError,'synthetic kernel error'):
                entry.run_kernel(physical,kernel,selected,refresh_class=FakeRefresh)
            self.assertIsNone(FakeRefresh.last())
            self.assertEqual(len(physical._inputs),1)
            self.assertIsNone(physical._outputs)
        finally:
            if was:gc.enable()

    def test_count_and_emit_fake_same_schedule_and_cleanup(self):
        kernel=DummyKernel();selected=planned(kernel);context1,context2=self.contexts()
        with context1,context2,patch.object(entry,'_api',return_value=api()):
            counted=entry.count_kernel(kernel,selected,metadata_template=['fixed'])
            emitted=entry.emit_kernel(kernel,selected,metadata_template=['fixed'],directory='synthetic-dir',
                frontend='synthetic-frontend',max_constant_bytes=4096,counted=counted)
        self.assertEqual(emitted['stats'],counted['stats'])
        self.assertFalse(emitted['native_compiled'])
        self.assertTrue(Physical.instances[-1].wrote)
        for physical in Physical.instances:
            self.assertEqual(physical._inputs,[]);self.assertIsNone(physical._outputs)

    def test_emit_count_mismatch_first_error_no_write_and_final_cleanup(self):
        kernel=DummyKernel();selected=planned(kernel);context1,context2=self.contexts()
        counted=dict(interface=dict(source=kernel.source()),stats=dict(counts={},ops_total=0,distinct_consts=0))
        with context1,context2,patch.object(entry,'_api',return_value=api()):
            with self.assertRaisesRegex(ValueError,'Count/Emit'):
                entry.emit_kernel(kernel,selected,metadata_template=[],directory='unused',frontend='unused',
                    max_constant_bytes=10,counted=counted)
        physical=Physical.instances[-1]
        self.assertFalse(physical.wrote);self.assertTrue(physical.closed)
        self.assertEqual(physical._inputs,[]);self.assertIsNone(physical._outputs)
        for limit in [True,0,float('nan')]:
            with self.assertRaises(ValueError):entry.emit_kernel(kernel,selected,metadata_template=[],
                directory='unused',frontend='unused',max_constant_bytes=limit,counted=counted)

    def test_emit_close_error_still_releases_physical_references(self):
        class BadClose(Physical):
            def close(self):self.closed=True;raise OSError('close failed')
        kernel=DummyKernel();selected=planned(kernel);context1,context2=self.contexts()
        counted=dict(interface=dict(source=kernel.source()),stats=dict(counts={},ops_total=0,distinct_consts=0))
        with context1,context2,patch.object(entry,'_api',return_value=dict(api(),Emit=BadClose)):
            with self.assertRaisesRegex(OSError,'close failed'):
                entry.emit_kernel(kernel,selected,metadata_template=[],directory='unused',frontend='unused',max_constant_bytes=10,counted=counted)
        self.assertEqual(Physical.instances[-1]._inputs,[]);self.assertIsNone(Physical.instances[-1]._outputs)

    def test_metadata_context_precedes_api_import_and_restores_on_failure(self):
        state={'entered':False};trace=[]
        @contextmanager
        def metadata(template):
            trace.append(('enter',template));state['entered']=True
            try:yield
            finally:state['entered']=False;trace.append(('exit',))
        def imported():
            self.assertTrue(state['entered']);raise RuntimeError('import rejected')
        with patch.object(entry,'_metadata',metadata),patch.object(entry,'_api',side_effect=imported):
            with self.assertRaises(RuntimeError):entry.count_kernel(DummyKernel(),planned(DummyKernel()),metadata_template=['recorded'])
        self.assertEqual(trace,[('enter',['recorded']),('exit',)])
        self.assertFalse(state['entered'])


if __name__=='__main__':unittest.main()
