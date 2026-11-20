"""Small metadata/dispatch checks; no model, official fit, init or compile runs."""

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import s1_orion_worker as worker
from pt_agent.experiments import r13_common_models
from pt_agent.experiments.r10_paper_metrics import StageRecorder, read_events, summarize_events


class WorkerTest(unittest.TestCase):
    def test_config_retains_native_parameters(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'configs').mkdir()
            base = dict(comment='native recipe', ckks_params=dict(LogN=16,
                LogQ=[55]+[40]*10, LogP=[61]*3, LogScale=40, H=192, RingType='standard'),
                boot_params=dict(LogP=[61]*8), orion=dict(margin=2, backend='lattigo',
                embedding_method='hybrid', fuse_modules=True, debug=True, io_mode='none',
                diags_path='../data/diagonals.h5', keys_path='../data/keys.h5'))
            (root / 'configs/resnet.yml').write_text(json.dumps(base))
            base['ckks_params']['LogQ'] = [55]+[40]*5
            base['orion']['margin'] = 1.5
            path = root / 'config.json'
            path.write_text(json.dumps(base))
            config, proof = worker.config_contract(SimpleNamespace(orion_root=root, config=path))
            self.assertEqual(config, base)
            self.assertFalse(proof['same_full_parameters'])
            self.assertEqual((proof['native_slots'], proof['l_eff'], proof['qwen48_input_level']),
                             (32768, 5, 8))
            base['ckks_params']['LogScale'] = 43
            path.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, 'retain resnet.yml'):
                worker.config_contract(SimpleNamespace(orion_root=root, config=path))

    def test_source_uses_only_inferred_input_metadata(self):
        from pt_agent.data import cifar
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {key:root/key for key in ('source','checkpoint','inference_preprocessing_source')}
            for path in paths.values():
                path.touch()
            factory = worker.FACTORY_PREFIX + worker.FACTORY_NAMES['resnet20']
            source = dict(factory=factory, path=factory, type='torch', spec_revision='R13',
                checkpoint=str(paths['checkpoint']), original_source=str(paths['source']),
                inference_preprocessing_source=str(paths['inference_preprocessing_source']),
                normalization=dict(mean=list(cifar.MEAN),std=list(cifar.STD)),
                preprocessing_provenance='official inference test', training_preprocessing_verified=False)
            split = dict(kind='legacy_cifar',heldout=list(cifar.HELDOUT),confirm=list(cifar.CONFIRM),
                report=list(cifar.REPORT),calibration='frozen training selection')
            record = root / 'model_info.json'
            # Deliberately missing graph directory: restoring the graph would fail.
            record.write_text(json.dumps(dict(source=source, split=split, graph='missing-graph',
                inputs=[dict(name='features',shape=[1,3,32,32],dtype='float64',kind='features')])))
            args = SimpleNamespace(model='resnet20',checkpoint=paths['checkpoint'],
                model_source=paths['source'],model_info=record)
            with patch.dict(r13_common_models.FACTORIES,{factory:paths},clear=True):
                name, info, proof = worker.source_contract(args)
                self.assertEqual(name, factory)
                self.assertEqual(info.graph.tensors['features'].shape,(1,3,32,32))
                self.assertFalse(proof['model_info_graph_loaded'])
                args.checkpoint = root / 'wrong'
                with self.assertRaisesRegex(ValueError, 'canonical model path'):
                    worker.source_contract(args)

    def test_domains_keep_official_centered_and_rule_endpoints_separate(self):
        activation = SimpleNamespace(degree=63,depth=7,input_min=-2.,input_max=4.,
            low=-3.5,high=5.5,prescale=2/9,constant=-2/9)
        net = SimpleNamespace(named_modules=lambda:[('activation',activation)])
        row = worker.activation_inventory(net)[0]
        self.assertEqual(row['native_domain'],[-3.5,5.5])
        self.assertEqual(row['rule63_endpoint_domain'],[-3.,6.])
        self.assertEqual(row['symmetric_maxabs_domain'],[-6.,6.])
        self.assertFalse(row['domain_equivalent'])
        self.assertTrue(row['depth_exceeds_l_eff'])
        self.assertEqual(row['symmetric_domain_covering_margin'],7/3)

    def test_native_count_is_reported_metadata_only(self):
        stream, log = io.StringIO(), io.StringIO()
        tee = worker.CompileOutput(stream, log)
        for piece in ('Preparing\n├── Network requires ', '123', ' bootstrap operations.\n'):
            tee.write(piece)
        tee.flush()
        self.assertEqual(tee.weighted_boots,123)
        self.assertEqual(stream.getvalue(),log.getvalue())
        self.assertIsNone(worker.CompileOutput(io.StringIO(),io.StringIO()).weighted_boots)

    def test_no_placement_requires_the_official_exception_origin(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace = {}
            # A controlled synthetic traceback, not the Orion implementation.
            code = compile("def finally_solve_full_level_dag():\n    raise ValueError('Automatic bootstrap placement failed. original detail')\n",
                           str(root/'orion/core/auto_bootstrap.py'),'exec')
            exec(code,namespace)
            try:
                namespace['finally_solve_full_level_dag']()
            except ValueError as exc:
                self.assertEqual(worker.classify_exception(exc,'compile',root),'no_placement')
                self.assertEqual(worker.classify_exception(exc,'fit',root),'driver_error')
            try:
                raise ValueError('Automatic bootstrap placement failed. original detail')
            except ValueError as exc:
                self.assertEqual(worker.classify_exception(exc,'compile',root),'driver_error')

    def test_error_clocks_close_and_preserve_native_phase_boundary(self):
        with TemporaryDirectory() as temporary:
            args = SimpleNamespace(work_dir=Path(temporary))
            result = dict(stage_clocks_ns={},fit_completed=1,compile_completed=0)
            recorder = StageRecorder(args.work_dir/'events.jsonl')
            with worker._stage(args,recorder,result,'fit'):
                pass
            with self.assertRaisesRegex(ValueError,'actual failure'):
                with worker._stage(args,recorder,result,'compile'):
                    raise ValueError('actual failure')
            clocks = result['stage_clocks_ns']
            self.assertEqual(result['fit_compile_seconds'],
                (clocks['fit']['elapsed_ns']+clocks['compile']['elapsed_ns'])/1e9)
            self.assertIsNone(result['init_scheme_seconds'])
            self.assertFalse(result['fit_compile_both_completed'])
            self.assertEqual(json.loads((args.work_dir/'phase_result.json').read_text()),
                             json.loads((args.work_dir/'worker_result.json').read_text()))
            summary = summarize_events(read_events(args.work_dir/'events.jsonl'))
            self.assertTrue(summary['attribution_ready'])
            self.assertEqual(summary['spans'][-1]['status'],'error')

    def test_main_failure_persists_original_trace_and_zero_crypto_observation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = ['--model','resnet20']
            for name in ('work-dir','checkpoint','model-source','model-info','records',
                         'selections','orion-root','config'):
                argv.extend(['--'+name,str(root/name)])
            def failure(args,recorder,result):
                result.update(active_stage='compile',compile_calls=1)
                raise RuntimeError('controlled driver failure')
            with patch.object(worker,'run',side_effect=failure):
                self.assertEqual(worker.main(argv),1)
            saved = json.loads((root/'work-dir/phase_result.json').read_text())
            self.assertEqual(saved['status'],'driver_error')
            self.assertEqual(saved['error'],'RuntimeError: controlled driver failure')
            self.assertEqual(saved['error_detail']['message'],'controlled driver failure')
            self.assertEqual(saved['compile_calls'],1)
            self.assertEqual(saved['actual_CKKS_executions'],0)
            self.assertIsNone(saved['observed_encrypted_bootstrap_calls'])
            self.assertIn('controlled driver failure',(root/'work-dir/traceback.log').read_text())
            self.assertIsNone(saved['pred_latency_s'])
            self.assertFalse(saved['native_compile_accepted'])

    def test_secondary_inventory_error_preserves_the_actual_compiler_exception(self):
        with TemporaryDirectory() as temporary:
            args = SimpleNamespace(work_dir=Path(temporary))
            result = {}
            primary = ValueError('original compiler placement error')
            with patch.object(worker,'_save_inventory',side_effect=RuntimeError('inventory error')):
                with self.assertRaises(ValueError) as caught:
                    try:
                        raise primary
                    finally:
                        worker._post_compile_inventory(args,result,None,primary)
                self.assertIs(caught.exception,primary)
                self.assertEqual(result['secondary_errors'][0]['message'],'inventory error')
                with self.assertRaisesRegex(RuntimeError,'inventory error'):
                    worker._post_compile_inventory(args,{},None,None)

    def test_open_compile_clock_is_durable_and_never_reports_fit_only_total(self):
        with TemporaryDirectory() as temporary:
            args = SimpleNamespace(work_dir=Path(temporary))
            result = dict(status='running',stage_clocks_ns=dict(
                fit=dict(started_ns=10,ended_ns=30,elapsed_ns=20,elapsed_s=20/1e9)))
            recorder = StageRecorder(args.work_dir/'events.jsonl')
            with worker._stage(args,recorder,result,'compile'):
                saved = json.loads((args.work_dir/'phase_result.json').read_text())
                self.assertIsNotNone(saved['stage_clocks_ns']['compile']['started_ns'])
                self.assertIsNone(saved['stage_clocks_ns']['compile']['ended_ns'])
                self.assertIsNone(saved['stage_clocks_ns']['compile']['elapsed_ns'])
                self.assertIsNone(saved['fit_compile_seconds'])
                self.assertEqual(saved['fit_compile_known_completed_seconds'],20/1e9)
                self.assertTrue(saved['fit_compile_clock_incomplete'])
            self.assertIsNotNone(result['fit_compile_seconds'])
            self.assertFalse(result['fit_compile_clock_incomplete'])


if __name__ == '__main__':
    unittest.main()
