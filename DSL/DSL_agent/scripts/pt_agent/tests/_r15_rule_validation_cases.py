"""R15 inner checks with temporary provenance, small graphs and numeric fakes."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class RuleValidationCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from pt_agent.experiments import r15_rule_validation_worker as worker
        from pt_agent.experiments.r13_common_models import FACTORIES
        from pt_agent.intake.graph import Graph, Node, TensorInfo
        from pt_agent.models.info import ModelInfo
        from pt_agent.data import cifar
        from pt_agent.dsl.parser import parse

        cls.np, cls.worker, cls.factories = np, worker, FACTORIES
        cls.Graph, cls.Node, cls.TensorInfo, cls.ModelInfo = Graph, Node, TensorInfo, ModelInfo
        cls.cifar, cls.parse = cifar, staticmethod(parse)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='r15-validation-cases-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def fixture(self, model=None):
        """Create synthetic prior records; never read a production model or dataset."""
        np, worker = self.np, self.worker
        model = model or next(iter(worker.MODEL_RECORDS))
        record, original = worker.MODEL_RECORDS[model], self.factories[model]
        shapes = {'camera_pixels': (1, 3, 32, 32), 'convolved': (1, 16, 32, 32),
                  'pooled': (1, 16), 'activated': (1, 16), 'declared_logits': (1, 10),
                  'stem.weight': (16, 3, 1, 1), 'head.weight': (10, 16), 'head.bias': (10,)}
        tensors = {name: self.TensorInfo(name, shape, 'input' if name == 'camera_pixels'
                                       else 'weight' if name.endswith(('weight', 'bias')) else 'act')
                   for name, shape in shapes.items()}
        graph = self.Graph('r15_fixture', [
            self.Node('conv', 'conv2d', ['camera_pixels'], ['convolved'],
                      dict(weight='stem.weight', stride=(1, 1), padding=(0, 0),
                           dilation=(1, 1), groups=1), 'fixture convolution'),
            self.Node('pool', 'mean', ['convolved'], ['pooled'],
                      dict(axes=(2, 3), keepdim=False), 'fixture mean'),
            self.Node('activation', 'act', ['pooled'], ['activated'], dict(fn='silu'), 'fixture SiLU'),
            self.Node('head', 'linear', ['activated'], ['declared_logits'],
                      dict(weight='head.weight', bias='head.bias'), 'fixture logits')],
            tensors, dict(**{'stem.weight': np.full((16, 3, 1, 1), 0.01, dtype=np.float64),
                             'head.weight': np.arange(160, dtype=np.float64).reshape(10, 16) / 100,
                             'head.bias': np.arange(10, dtype=np.float64) / 1000}),
            ['camera_pixels'], ['declared_logits'])
        source = dict(type='torch', path=model, factory=model, spec_revision='R13',
                      original_source=str(original['source'].resolve()),
                      checkpoint=str(original['checkpoint'].resolve()),
                      inference_preprocessing_source=str(original['inference_preprocessing_source'].resolve()),
                      normalization=dict(mean=list(self.cifar.MEAN), std=list(self.cifar.STD)),
                      preprocessing_provenance='official inference test', training_preprocessing_verified=False,
                      intake=dict(original_nodes=4, folded_nodes=0, original_unmapped={}))
        split = dict(kind='legacy_cifar', heldout=list(self.cifar.HELDOUT),
                     confirm=list(self.cifar.CONFIRM), report=list(self.cifar.REPORT),
                     calibration='frozen training selection')
        info = self.ModelInfo(source, graph,
                              dict(mode='original tensors', internal_regression_source='legacy_cifar'),
                              [dict(name='camera_pixels', kind='features', shape=[1, 3, 32, 32], dtype='float64')],
                              None, 'classification', 'classification_top1', dict(acc_drop_pp=0.5), split,
                              [dict(task_inference='fixture classification')])
        root = self.directory / model.rsplit(':', 1)[-1]
        stem=model.rsplit(':',1)[-1].replace('_','-')
        r13_dir=root/('r13-common-'+stem+'-20261115')
        rule_dir=root/('r14-common-'+stem+'-20261115')
        model_info, rule_work = r13_dir / 'worker/model_info', rule_dir / 'worker'
        rule_work.mkdir(parents=True)
        info.save(model_info)
        r13 = dict(round='R13', mode='common-model-intake-preparation', status='ok', model=model,
                   worker=dict(round='R13', status='ok', model=model, model_source=copy.deepcopy(source),
                               split=copy.deepcopy(split), import_completed=True, real_probes_passed=True,
                               graph_math_passed=True, roundtrip_passed=True, static_coverage=True,
                               artifacts=dict(model_info=str(model_info))))
        r13_path = r13_dir / 'result.json'
        r13_path.write_text(json.dumps(r13))
        text = ('@fhe.program(slots=16384, refresh="compiler", placement="depth-dp")\n'
                'def rule(camera_pixels: Cipher[1,3,32,32]):\n'
                '    p = pack(camera_pixels, layout="mux", gap=1)\n'
                '    c = conv2d(p, w=W["stem.weight"], stride=(1,1), pad=(0,0), algo="mpcb")\n'
                '    m = mean(c, axis=(2,3))\n'
                '    a = act(m, fn="silu", method="cheb", deg=63, domain=(-2.0,2.0))\n'
                '    y = linear(a, w=W["head.weight"], b=W["head.bias"], algo="bsgs")\n'
                '    return y\n')
        program, count = rule_work / 'program.ptdsl', rule_work / 'count.json'
        program.write_text(text)
        stats = dict(counts=dict(add=2, sub=0, neg=0, mul_cc=1, mul_cp=1, rot=0, boot=0),
                     ops_total=4, distinct_consts=2)
        count.write_text(json.dumps(dict(stats=stats, output_layouts=[])))
        rule_options = dict(deg=63, margin=1.25, slots=16384, placement='depth-dp',
                            refresh='compiler', act_method='cheb')
        rule_worker = dict(round='R14', model=model, status='ok' if record['round'] == 'R14'
                           else 'artifact_budget_exceeded', failed_phase=None if record['round'] == 'R14'
                           else 'emit_write', rule_options=rule_options,
                           count_stats=copy.deepcopy(stats), emit_stats=copy.deepcopy(stats),
                           artifacts=dict(program=str(program), count=str(count)),
                           source_checked=True, reference_passed=True, calibration_completed=True,
                           plan_completed=True, typecheck_passed=True, signature_passed=True,
                           count_completed=True, stats_equal=True,
                           emit_completed=True if record['round'] == 'R14' else None,
                           lowering_ready=True if record['round'] == 'R14' else None,
                           r13_source_result=str(r13_path),
                           calibration_attempts=1, plan_attempts=1, count_attempts=1, emit_attempts=1)
        rule_result = dict(round='R14', mode='common-model-rule-preparation', model=model,
                           status=rule_worker['status'], worker=rule_worker,
                           source_r13=dict(run=str(r13_dir), result=str(r13_path),
                                           artifacts=dict(model_info=str(model_info))),
                           timing=dict(spans=[dict(event_id='old:calibration', stage_id='calibrate',
                                                  inclusive_ns=123, exclusive_ns=123)]))
        rule_path = rule_dir / 'result.json'
        if record['round'] == 'R14':
            exported, export_dir = rule_result, rule_dir
        else:
            export_dir = root / record['export_run']
            export_dir.mkdir()
            export_worker = dict(round=record['round'], model=model, status='ok',
                                  typecheck_passed=True, signature_passed=True, stats_equal=True,
                                  emit_completed=True, lowering_ready=True, rule_options=rule_options,
                                  count_stats=copy.deepcopy(stats), emit_stats=copy.deepcopy(stats),
                                  source_checked=True,
                                  source_resume=dict(source_result=str(rule_path), program=str(program),
                                                     count=str(count), r13_result=str(r13_path),
                                                     model_info=str(model_info)), artifacts={})
            exported = dict(round=record['round'], mode=record['mode'], status='ok', model=model,
                            worker=export_worker, source_r14=dict(result=str(rule_path), run=str(rule_dir),
                            artifacts=dict(source_result=str(rule_path), program=str(program), count=str(count),
                                           r13_result=str(r13_path), model_info=str(model_info))))
        emit = export_dir / 'worker/emit'
        emit.mkdir(parents=True)
        trace, constants = emit / 'candidate_trace.py', emit / 'constants.npz'
        trace.write_text("raise AssertionError('R15 must not execute generated source')\n")
        constants.write_text('This fixture archive must never be loaded by R15.\n')
        exported['worker']['artifacts'].update(trace_source=str(trace), constants=str(constants))
        rule_path.write_text(json.dumps(rule_result))
        export_path = export_dir / 'result.json'
        export_path.write_text(json.dumps(exported))
        records = root / 'records'
        records.mkdir()
        (records / 'dataset.json').write_text(json.dumps(self.dataset_metadata()))
        selected = dict(export_result=export_path, rule_result=rule_path, program=program,
                        count=count, r13_result=r13_path, model_info=model_info,
                        records=records, work_dir=root / 'validation/worker')
        return SimpleNamespace(model=model, info=info, r13=r13, exported=exported,
                               rule=rule_result, stats=stats, text=text, paths=selected,
                               args=SimpleNamespace(**selected),
                               cli=sum((['--'+key.replace('_','-'), str(value)]
                                        for key,value in selected.items()), []))

    def dataset_metadata(self):
        return dict(format='poseidon-cifar10-binary-records-v1', record_bytes=3073,
                    row_label_byte=0, preprocessing_applied=False, train_count=50000,
                    test_count=10000, pixel_layout='uint8 planar RGB 3x32x32, C order')

    def fixture_model_records(self, fixture):
        records = copy.deepcopy(self.worker.MODEL_RECORDS)
        records[fixture.model]['act_count'] = 1
        return records

    def test_math_uses_the_original_elementwise_formula(self):
        np = self.np
        reference = np.array([[0., 1., -2., 1e6, 0., 0., 0., 0., 0., 0.]], dtype=np.float64)
        observed = reference + (1e-12 + 1e-12*np.abs(reference))*0.5
        checked = self.worker.check_math(reference, observed)
        self.assertTrue(checked['passed'])
        self.assertEqual(checked['criterion'], '1e-12+1e-12*abs(reference)')
        self.assertLessEqual(checked['max_error_ratio'], 1)
        self.assertEqual(checked['reference_shape'], [1, 10])

    def test_math_keeps_the_boundary_and_rejects_the_next_float(self):
        np = self.np
        reference = np.zeros((1,10), dtype=np.float64)
        observed = reference.copy(); observed[0,0] = 1e-12
        self.assertTrue(self.worker.check_math(reference, observed)['passed'])
        observed[0,0] = np.nextafter(1e-12, np.inf)
        checked = self.worker.check_math(reference, observed)
        self.assertFalse(checked['passed'])
        self.assertGreater(checked['max_error_ratio'], 1)

    def test_math_rejects_shape_and_original_dtype_before_conversion(self):
        np = self.np
        valid = np.zeros((1,10), dtype=np.float64)
        for reference, observed in ((valid, valid.ravel()), (valid, valid.astype(np.float32)),
                                    (valid.astype(np.float32), valid)):
            with self.subTest(reference_dtype=reference.dtype, observed_shape=observed.shape):
                self.assertFalse(self.worker.check_math(reference, observed)['passed'])

    def test_math_rejects_nonfinite_values_on_either_side(self):
        np = self.np
        for bad in (np.nan, np.inf, -np.inf):
            for side in ('reference','observed'):
                with self.subTest(bad=bad, side=side):
                    reference = np.zeros((1,10), dtype=np.float64); observed = reference.copy()
                    (reference if side=='reference' else observed)[0,0] = bad
                    checked = self.worker.check_math(reference, observed)
                    self.assertFalse(checked['passed'])
                    self.assertFalse(checked['finite'])

    def test_all_three_export_sources_preserve_the_original_features_and_policy(self):
        for model in self.worker.MODEL_RECORDS:
            with self.subTest(model=model):
                fixture = self.fixture(model)
                before = copy.deepcopy((fixture.info.source, fixture.info.inputs, fixture.info.split,
                                        fixture.info.task, fixture.info.metric, fixture.info.threshold))
                checked = self.worker.validate_export_source(
                    fixture.exported, fixture.rule, fixture.r13, fixture.info)
                self.assertTrue(checked['passed'], checked)
                self.assertEqual((fixture.info.source, fixture.info.inputs, fixture.info.split,
                                  fixture.info.task, fixture.info.metric, fixture.info.threshold), before)
                self.assertEqual(fixture.info.inputs[0]['kind'], 'features')
                self.assertEqual(fixture.info.threshold, dict(acc_drop_pp=0.5))

    def test_source_rejects_changed_policy_features_split_or_prior_success(self):
        fixture = self.fixture()
        for field in ('task','metric','threshold','features','split','source_status','prior_reference'):
            with self.subTest(field=field):
                exported, rule, r13, info = copy.deepcopy(
                    (fixture.exported, fixture.rule, fixture.r13, fixture.info))
                if field=='task': info.task='regression'
                elif field=='metric': info.metric='agreement'
                elif field=='threshold': info.threshold=dict(acc_drop_pp=1.0)
                elif field=='features': info.inputs[0]['kind']='image'
                elif field=='split': info.split['report']=[]
                elif field=='source_status': exported['status']='error'
                else: rule['worker']['reference_passed']=False
                self.assertFalse(self.worker.validate_export_source(exported,rule,r13,info)['passed'])

    def test_source_paths_only_observe_exported_archive_and_source_metadata(self):
        for model in self.worker.MODEL_RECORDS:
            with self.subTest(model=model):
                fixture=self.fixture(model)
                paths=self.worker._source_paths(fixture.args,fixture.exported,fixture.rule,fixture.r13)
                for name in ('trace_source','constants'):
                    saved=paths['exported_artifacts'][name]
                    path=Path(fixture.exported['worker']['artifacts'][name])
                    self.assertEqual(saved,dict(path=str(path.resolve()),bytes=path.stat().st_size))
                self.assertFalse(paths['included_in_current_turnaround'])
                changed=copy.deepcopy(fixture.rule)
                changed['worker']['r13_source_result']=str(self.directory/'other-result.json')
                with self.assertRaises(self.worker.WorkerFailure) as caught:
                    self.worker._source_paths(fixture.args,fixture.exported,changed,fixture.r13)
                self.assertEqual(caught.exception.status,'source_failed')

    def test_original_rule_header_strategy_and_activation_count_are_required(self):
        fixture=self.fixture()
        program=self.parse(fixture.text)
        with patch.object(self.worker,'MODEL_RECORDS',self.fixture_model_records(fixture)):
            self.assertTrue(self.worker._check_program(fixture.info,program)['passed'])
            for case in ('slots','placement','degree','method','domain','count'):
                with self.subTest(case=case):
                    changed=copy.deepcopy(program)
                    act=next(op for op in changed.ops if op.kind=='act')
                    if case=='slots':changed=replace(changed,header=replace(changed.header,slots=8192))
                    elif case=='placement':changed=replace(changed,header=replace(changed.header,placement='greedy'))
                    elif case=='degree':act.attrs['deg']=31
                    elif case=='method':act.attrs['method']='newton'
                    elif case=='domain':act.attrs['domain']=(2.,2.)
                    else:changed=replace(changed,ops=tuple(op for op in changed.ops if op.kind!='act'))
                    self.assertFalse(self.worker._check_program(fixture.info,changed)['passed'])

    def observer_program(self):
        return self.parse('@fhe.program(slots=16384, refresh="compiler", placement="depth-dp")\n'
                          'def rule(x: Cipher[1,3,32,32]):\n'
                          '    p = pack(x, layout="mux", gap=1)\n'
                          '    a = act(p, fn="silu", method="cheb", deg=63, domain=(-2.0,2.0))\n'
                          '    b = act(a, fn="silu", method="cheb", deg=63, domain=(-4.0,4.0))\n'
                          '    return b\n')

    def test_observer_keeps_distinct_op_ids_sample_counts_and_linear_point_nine(self):
        program = self.observer_program()
        acts = [op for op in program.ops if op.kind=='act']
        rows = self.worker._domain_rows(program)
        for sample_id in ('test:9000','test:9001'):
            callback, finish = self.worker._tensor_observer(program,sample_id,rows)
            callback(acts[0].inputs[0], self.np.array([-1.8,1.8]), ('cheb',63,-2.,2.))
            callback(acts[1].inputs[0], self.np.array([-3.6,3.6]), ('cheb',63,-4.,4.))
            finish()
        self.assertEqual([row['op_id'] for row in rows], [op.id for op in acts])
        self.assertEqual([row['count'] for row in rows], [2,2])
        self.assertTrue(all(abs(row['max_u']-0.90)<1e-15 for row in rows))

    def test_observer_rejects_wrong_order_name_cfg_missing_and_extra_calls(self):
        program=self.observer_program(); acts=[op for op in program.ops if op.kind=='act']
        for case in ('order','name','method','degree','domain','missing','extra'):
            with self.subTest(case=case):
                callback,finish=self.worker._tensor_observer(program,'test:9000',self.worker._domain_rows(program))
                with self.assertRaises(self.worker.WorkerFailure) as caught:
                    if case=='missing': finish()
                    else:
                        name=acts[1].inputs[0] if case=='order' else 'different' if case=='name' else acts[0].inputs[0]
                        cfg=('newton',63,-2.,2.) if case=='method' else ('cheb',31,-2.,2.) if case=='degree' else ('cheb',63,-3.,3.) if case=='domain' else ('cheb',63,-2.,2.)
                        callback(name,self.np.array([0.]),cfg)
                        if case=='extra':
                            callback(acts[1].inputs[0],self.np.array([0.]),('cheb',63,-4.,4.))
                            callback(acts[1].inputs[0],self.np.array([0.]),('cheb',63,-4.,4.))
                self.assertEqual(caught.exception.status,'observer_failed')

    def test_observer_rejects_nonfinite_activations(self):
        program=self.observer_program(); act=next(op for op in program.ops if op.kind=='act')
        for value in (self.np.nan,self.np.inf):
            with self.subTest(value=value):
                callback,_=self.worker._tensor_observer(program,'test:9000',self.worker._domain_rows(program))
                with self.assertRaises(self.worker.WorkerFailure) as caught:
                    callback(act.inputs[0],self.np.array([value]),('cheb',63,-2.,2.))
                self.assertEqual(caught.exception.status,'nonfinite')

    def fake_sample_batch(self, fixture):
        np=self.np
        images=np.broadcast_to(np.arange(1000,dtype=np.float64).reshape(1000,1,1,1),
                               (1000,3,32,32))
        source=dict(split='heldout', indices=list(range(9000,10000)),
                    sample_ids=['test:'+str(i) for i in range(9000,10000)],
                    labels=[0]*1000, count=1000, batch=1,
                    normalization=copy.deepcopy(fixture.info.source['normalization']))
        return {'camera_pixels':images},np.zeros(1000,dtype=np.int64),source

    def test_heldout_source_requests_only_the_fixed_thousand_and_original_normalization(self):
        from pt_agent.data import samples

        fixture=self.fixture()
        arrays,labels,_=self.fake_sample_batch(fixture)
        source=Mock()
        source.get.return_value=(arrays,labels)
        with patch.object(self.cifar,'RECORDS',fixture.args.records), \
             patch.object(samples,'LegacyCifar',return_value=source) as factory:
            actual_images,actual_labels,metadata=self.worker._read_heldout(fixture.args,fixture.info)
        factory.assert_called_once_with(fixture.info.graph,normalization=(
            fixture.info.source['normalization']['mean'],fixture.info.source['normalization']['std']))
        source.get.assert_called_once_with('heldout',1000)
        self.assertIs(actual_images,arrays)
        self.assertIs(actual_labels,labels)
        self.assertEqual(metadata['indices'],list(range(9000,10000)))
        self.assertEqual(metadata['sample_ids'],['test:'+str(i) for i in range(9000,10000)])
        self.assertEqual(metadata['source_split'],'test')
        self.assertEqual(metadata['shard'],'test-09000.records')
        self.assertEqual(metadata['batch'],1)
        self.assertEqual(metadata['confirm_samples'],0)
        self.assertEqual(metadata['report_samples'],0)

    def test_heldout_source_rejects_wrong_origin_metadata_shape_dtype_labels_and_nonfinite(self):
        from pt_agent.data import samples

        fixture=self.fixture()
        arrays,labels,_=self.fake_sample_batch(fixture)
        metadata_path=fixture.args.records/'dataset.json'
        for case in ('origin','metadata','shape','dtype','labels','label_range','nonfinite'):
            with self.subTest(case=case):
                metadata_path.write_text(json.dumps(self.dataset_metadata()))
                test_arrays,test_labels=arrays,labels
                if case=='metadata':
                    metadata=self.dataset_metadata();metadata['preprocessing_applied']=True
                    metadata_path.write_text(json.dumps(metadata))
                elif case=='shape':test_arrays={'camera_pixels':arrays['camera_pixels'][:999]}
                elif case=='dtype':test_arrays={'camera_pixels':self.np.broadcast_to(
                    self.np.array(0.,dtype=self.np.float32),(1000,3,32,32))}
                elif case=='labels':test_labels=labels.astype(self.np.float64)
                elif case=='label_range':test_labels=self.np.full(1000,10,dtype=self.np.int64)
                elif case=='nonfinite':test_arrays={'camera_pixels':self.np.broadcast_to(
                    self.np.array(self.np.nan),(1000,3,32,32))}
                source=Mock();source.get.return_value=(test_arrays,test_labels)
                canonical=fixture.args.records if case!='origin' else self.directory/'elsewhere'
                with patch.object(self.cifar,'RECORDS',canonical), \
                     patch.object(samples,'LegacyCifar',return_value=source), \
                     self.assertRaises(self.worker.WorkerFailure) as caught:
                    self.worker._read_heldout(fixture.args,fixture.info)
                self.assertEqual(caught.exception.status,'nonfinite' if case=='nonfinite'
                                 else 'source_failed' if case in ('origin','metadata') else 'shape_failed')
                if case in ('origin','metadata'):source.get.assert_not_called()

    def test_packed_helper_streams_constants_copies_logits_and_releases_on_failure(self):
        from pt_agent import backends, lower

        image=self.np.zeros((1,3,32,32),dtype=self.np.float64)
        output=self.np.zeros((1,10),dtype=self.np.float64)
        for failure in (False,True):
            with self.subTest(failure=failure):
                backend=Mock(outputs=['packed output'])
                packing=object();layout=object();program=object();weights={}
                with patch.object(backends,'MockBackend',return_value=backend) as factory, \
                     patch.object(lower,'pack_inputs',return_value=packing) as pack, \
                     patch.object(lower,'lower',side_effect=RuntimeError('fixture lowering failure')
                                  if failure else None,return_value=({},layout)) as lowering, \
                     patch.object(lower,'decode_outputs',return_value=[output]) as decode:
                    if failure:
                        with self.assertRaisesRegex(RuntimeError,'fixture lowering failure'):
                            self.worker._packed_logits(program,weights,image,act_mode='program')
                    else:
                        observed=self.worker._packed_logits(program,weights,image,act_mode='exact')
                        self.np.testing.assert_array_equal(observed,output)
                        self.assertFalse(self.np.shares_memory(observed,output))
                        decode.assert_called_once_with(backend.outputs,layout)
                factory.assert_called_once_with(slots=16384,batch=1,stream_constants=True)
                pack.assert_called_once_with(program,[image])
                backend.bind.assert_called_once_with(packing)
                lowering.assert_called_once_with(program,weights,backend,
                                                  act_mode='program' if failure else 'exact')
                backend.release_batch.assert_called_once_with()

    def invoke(self, fixture, *, flips=0, activation=0.2, exact_error=False,
               program_math_error=False, runtime_error_index=None,
               nonfinite_index=None, skip_observer=False):
        """Exercise worker orchestration using scalar protocol engines only."""
        from pt_agent import analyzer, backends, tensor_sim, toolchain
        from pt_agent.planners import rule

        np=self.np
        calls=[]
        tracked={name:[] for name in self.worker.ENGINE_NAMES}
        write_json=self.worker.write_json
        def write_fixture_json(path,value):
            # Numeric protocol loops keep the full final result. Intermediate
            # checkpoint serialization is covered separately with a small case.
            if Path(path).name=='worker_result.json':
                boundaries=value.get('case_budget_observations',[])
                if not boundaries or boundaries[-1].get('boundary')!='terminal':
                    return
            write_json(path,value)
        def index(image):
            return int(np.asarray(image)[0,0,0,0])
        def logits(i,mode):
            value=np.zeros((1,10),dtype=np.float64)
            if mode=='program' and i<flips:value[0,1]=3.
            else:value[0,0]=2.25 if mode=='program' else 2.
            return value
        def reference(graph,inputs,*,keep=False):
            self.assertFalse(keep)
            i=index(inputs[graph.inputs[0]])
            calls.append((i,'reference_forward'));tracked['reference_forward'].append(i)
            return {graph.outputs[0]:logits(i,'exact')}
        def tensor(program,weights,image,*,cache,act_mode,observer=None,**kwargs):
            self.assertFalse(cache)
            i=index(image);engine='tensor_'+act_mode
            calls.append((i,engine));tracked[engine].append(i)
            if act_mode=='program' and i==runtime_error_index:
                raise RuntimeError('fixture program execution failed')
            if act_mode=='program' and not skip_observer:
                self.assertIsNotNone(observer)
                for op in program.ops:
                    if op.kind=='act':
                        lo,hi=op.attrs['domain']
                        observer(op.inputs[0],np.array([activation],dtype=np.float64),
                                 ('cheb',63,lo,hi))
            value=logits(i,act_mode)
            if act_mode=='exact' and exact_error:value[0,0]+=1e-6
            if act_mode=='program' and i==nonfinite_index:value[0,0]=np.nan
            return value
        def packed(program,weights,image,*,act_mode):
            i=index(image);engine='mock_'+act_mode
            calls.append((i,engine));tracked[engine].append(i)
            value=logits(i,act_mode)
            if act_mode=='program' and program_math_error:value[0,0]+=1e-6
            return value
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.worker,'write_json',side_effect=write_fixture_json))
            stack.enter_context(patch.object(self.worker,'MODEL_RECORDS',
                                             self.fixture_model_records(fixture)))
            stack.enter_context(patch.object(self.worker,'_initialize_torch'))
            read_samples=stack.enter_context(patch.object(self.worker,'_read_heldout',
                                                          return_value=self.fake_sample_batch(fixture)))
            stack.enter_context(patch.object(analyzer,'reference_forward',side_effect=reference))
            stack.enter_context(patch.object(tensor_sim,'run_tensor',side_effect=tensor))
            stack.enter_context(patch.object(self.worker,'_packed_logits',side_effect=packed))
            forbidden=[]
            for target,name in ((analyzer,'calibrate'),(rule,'plan'),
                                (backends,'CountBackend'),(backends,'EmitBackend'),
                                (toolchain,'trace'),(toolchain,'compile'),(toolchain,'execute_cpu')):
                forbidden.append(stack.enter_context(patch.object(
                    target,name,side_effect=AssertionError('Forbidden stage: '+name))))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            code=self.worker.main(fixture.cli)
            for denied in forbidden:denied.assert_not_called()
        result=json.loads((fixture.paths['work_dir']/'worker_result.json').read_text())
        return code,result,tracked,calls,read_samples

    def test_small_phase_writes_real_budget_observations_and_partial_result_on_error(self):
        work=self.directory/'small-phase/worker'
        work.mkdir(parents=True)
        args=SimpleNamespace(work_dir=work)
        result=self.worker._initial_result()
        recorder=self.worker.StageRecorder(work/'events.jsonl')
        with self.assertRaisesRegex(self.worker.WorkerFailure,'fixture phase failure'):
            with self.worker._phase(args,recorder,result,'fixture_phase'):
                (work/'partial.txt').write_text('saved partial observation\n')
                raise self.worker.WorkerFailure('fixture phase failure',status='observer_failed')
        saved=json.loads((work/'worker_result.json').read_text())
        observations=saved['case_budget_observations']
        self.assertEqual([item['boundary'] for item in observations],['before','after'])
        self.assertTrue(all(item['limit_bytes']==4*1024**3 for item in observations))
        self.assertTrue(all(item['within_budget'] for item in observations))
        self.assertGreater(observations[-1]['bytes'],observations[0]['bytes'])
        self.assertFalse(saved['evaluation_completed'])
        self.assertIsNone(saved['validation_metrics'])
        self.assertTrue((work/'partial.txt').is_file())

    def test_full_fake_run_reuses_three_probes_and_never_runs_forbidden_stages(self):
        fixture=self.fixture()
        code,result,tracked,calls,read_samples=self.invoke(fixture)
        self.assertEqual(code,0,result.get('error'))
        self.assertEqual(result['status'],'ok')
        self.assertTrue(result['evaluation_completed'])
        self.assertTrue(result['heldout_acceptance'])
        self.assertEqual(result['completed_samples'],1000)
        self.assertEqual(tracked['reference_forward'],list(range(1000)))
        self.assertEqual(tracked['tensor_program'],list(range(1000)))
        for name in ('tensor_exact','mock_exact','mock_program'):
            self.assertEqual(tracked[name],[0,1,2])
        expected=[]
        for i in range(3):
            expected.extend((i,name) for name in ['reference_forward','tensor_exact','mock_exact',
                                                'tensor_program','mock_program'])
        self.assertEqual(calls[:15],expected)
        for engine,expected_count in {'reference_forward':1000,'tensor_program':1000,
                                      'tensor_exact':3,'mock_exact':3,'mock_program':3}.items():
            self.assertEqual(result['engine_counts'][engine],
                             dict(attempted=expected_count,completed=expected_count))
        self.assertEqual(result['domain_rows'][0]['count'],1000)
        self.assertFalse(result['candidate_frozen'])
        for field in self.worker.ZERO_FIELDS:self.assertEqual(result[field],0)
        for field in self.worker.UNMEASURED_FIELDS:self.assertIsNone(result[field])
        self.assertEqual(result['source_count_stats'],fixture.stats)
        read_samples.assert_called_once()
        self.assertGreater(result['max_logit_absolute_difference'],1e-12)

    def test_half_percentage_point_drop_passes_and_larger_drop_is_normal_rejection(self):
        for flips,accepted in ((5,True),(6,False)):
            with self.subTest(flips=flips):
                fixture=self.fixture()
                code,result,_,_,_=self.invoke(fixture,flips=flips)
                self.assertEqual(code,0,result.get('error'))
                self.assertEqual(result['status'],'ok')
                self.assertTrue(result['evaluation_completed'])
                self.assertIs(result['task_passed'],accepted)
                self.assertIs(result['heldout_acceptance'],accepted)
                metrics=result['validation_metrics']
                self.assertEqual(metrics['exact_top1'],100.)
                self.assertEqual(metrics['top1'],100.-flips/10)
                self.assertAlmostEqual(metrics['acc_drop_pp'],flips/10)
                self.assertFalse(result['candidate_frozen'])
                # A fresh, explicitly named fixture case is used for each policy observation.
                self.directory=self.directory/('after-'+str(flips))
                self.directory.mkdir()

    def test_finite_domain_rejection_completes_all_samples_without_rewriting(self):
        fixture=self.fixture()
        code,result,tracked,_,_=self.invoke(fixture,activation=1.82)
        self.assertEqual(code,0,result.get('error'))
        self.assertEqual(result['status'],'ok')
        self.assertTrue(result['evaluation_completed'])
        self.assertTrue(result['task_passed'])
        self.assertFalse(result['domain_passed'])
        self.assertFalse(result['heldout_acceptance'])
        self.assertAlmostEqual(result['max_u'],0.91)
        self.assertEqual(len(tracked['tensor_program']),1000)
        self.assertEqual(fixture.paths['program'].read_text(),fixture.text)

    def assert_partial(self,result):
        self.assertFalse(result['evaluation_completed'])
        self.assertFalse(result['candidate_frozen'])
        for field in ('task_passed','domain_passed','heldout_acceptance','validation_metrics'):
            self.assertIsNone(result[field])
        for field in self.worker.UNMEASURED_FIELDS:self.assertIsNone(result[field])
        self.assertEqual(result['confirm_samples'],0)
        self.assertEqual(result['report_samples'],0)

    def test_exact_probe_mismatch_stops_before_later_engines_and_keeps_partial_nulls(self):
        fixture=self.fixture()
        code,result,tracked,_,_=self.invoke(fixture,exact_error=True)
        self.assertEqual(code,1)
        self.assertEqual(result['status'],'math_failed')
        self.assertEqual(tracked['reference_forward'],[0])
        self.assertEqual(tracked['tensor_exact'],[0])
        self.assertEqual(tracked['mock_exact'],[])
        self.assertEqual(tracked['tensor_program'],[])
        self.assertEqual(result['completed_samples'],0)
        self.assert_partial(result)

    def test_program_probe_uses_packed_program_reference_and_stops_on_mismatch(self):
        fixture=self.fixture()
        code,result,tracked,calls,_=self.invoke(fixture,program_math_error=True)
        self.assertEqual(code,1)
        self.assertEqual(result['status'],'math_failed')
        self.assertEqual(tracked['mock_program'],[0])
        self.assertEqual(tracked['tensor_program'],[0])
        self.assertEqual(result['completed_samples'],1)
        self.assertEqual(len(calls),5)
        self.assert_partial(result)

    def test_runtime_error_after_three_valid_probes_preserves_actual_counts(self):
        fixture=self.fixture()
        code,result,tracked,_,_=self.invoke(fixture,runtime_error_index=3)
        self.assertEqual(code,1)
        self.assertNotEqual(result['status'],'ok')
        self.assertEqual(result['completed_samples'],3)
        self.assertEqual(result['engine_counts']['reference_forward'],dict(attempted=4,completed=4))
        self.assertEqual(result['engine_counts']['tensor_program'],dict(attempted=4,completed=3))
        self.assertEqual(tracked['mock_program'],[0,1,2])
        self.assert_partial(result)

    def test_nonfinite_program_output_is_an_execution_failure_with_partial_metrics(self):
        fixture=self.fixture()
        code,result,tracked,_,_=self.invoke(fixture,nonfinite_index=3)
        self.assertEqual(code,1)
        self.assertEqual(result['status'],'nonfinite')
        self.assertEqual(result['completed_samples'],3)
        self.assertEqual(tracked['tensor_program'],[0,1,2,3])
        self.assert_partial(result)

    def test_missing_activation_observation_stops_before_program_packed_probe(self):
        fixture=self.fixture()
        code,result,tracked,_,_=self.invoke(fixture,skip_observer=True)
        self.assertEqual(code,1)
        self.assertEqual(result['status'],'observer_failed')
        self.assertEqual(tracked['tensor_program'],[0])
        self.assertEqual(tracked['mock_program'],[])
        self.assert_partial(result)

    def test_invalid_export_source_prevents_sample_reads_and_numeric_calls(self):
        fixture=self.fixture()
        fixture.exported['worker']['emit_completed']=None
        fixture.paths['export_result'].write_text(json.dumps(fixture.exported))
        code,result,tracked,_,read_samples=self.invoke(fixture)
        self.assertEqual(code,1)
        self.assertEqual(result['status'],'source_failed')
        self.assertTrue(all(not calls for calls in tracked.values()))
        read_samples.assert_not_called()
        self.assert_partial(result)
