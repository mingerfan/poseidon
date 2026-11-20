"""New pricing policy and private outer seam; no real model or compiler."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import a_group_profile_sum_policy as p
from pt_agent.experiments import cold_rule63_paper_policy as w
from pt_agent.experiments import cold_rule63_whole_model as original


class PaperPolicy(unittest.TestCase):
    def test_only_policy_relinearize_is_excluded_other_unknowns_keep_null(self):
        prediction=dict(table_known_sum_us=1250000,missing=[dict(index=1,op='relinearize'),dict(index=7,op='relinearize')],by_op_us=dict(boot=1250000))
        unchanged=json.dumps(prediction,sort_keys=True);found=p.profile_sum(prediction)
        self.assertEqual(found['pred_latency_s'],1.25);self.assertEqual(found['excluded_counts'],dict(relinearize=2))
        self.assertEqual(found['missing_remaining'],[]);self.assertEqual(json.dumps(prediction,sort_keys=True),unchanged)
        prediction['missing'].append(dict(index=8,op='unknown_operation',level=4))
        found=p.profile_sum(prediction);self.assertIsNone(found['pred_latency_s'])
        self.assertEqual(found['known_cost_partial_s'],1.25)
        self.assertEqual(found['missing_remaining'],[dict(index=8,op='unknown_operation',level=4)])
        for invalid in(None,{},dict(table_known_sum_us=1),dict(table_known_sum_us=float('nan'),missing=[])):
            self.assertIsNone(p.profile_sum(invalid)['pred_latency_s'])

    def test_execute_only_private_paper_and_snapshot_bindings_change(self):
        private=w.execute.__globals__
        self.assertIsNot(private,original.__dict__);self.assertIs(w.execute.__code__,original.execute.__code__)
        self.assertEqual(set(private),set(original.__dict__))
        for key,value in original.__dict__.items():
            if key not in('paper_record','snapshot'):self.assertIs(private[key],value,key)
        self.assertIs(private['paper_record'],w.paper_record);self.assertIs(private['snapshot'],w.snapshot)
        self.assertIs(private['phase_command'],original.phase_command)
        self.assertIsNot(original.paper_record,w.paper_record)

    def test_real_private_outer_fake_workers_publish_new_sum_and_keep_original_workers(self):
        from pt_agent import paths,registry
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);args=SimpleNamespace(model='resnet20_silu',placement='depth',out=root/'case',
                paper_runs=root/'runs.jsonl',inputs=root/'input',selections=root/'selections',frontend=root/'frontend',
                profile=root/'profile.json',metadata_template=root/'template',seconds=12000,case_gib=64,
                diagnostic_elide_final_ir=True)
            def observation():return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[],started_ns=100,ended_ns=100000100)
            commands=[]
            def worker(argv,**kwargs):
                commands.append(argv);phase=argv[argv.index('--worker-phase')+1]
                self.assertIn('pt_agent.experiments.cold_rule63_whole_model',argv)
                self.assertNotIn('pt_agent.experiments.cold_rule63_paper_policy',argv)
                row=dict(status='ok')
                if phase=='compile':row.update(native_process=observation(),compiled_bootstraps=230,
                    execution_steps=10,execution_ops=dict(boot=230,relinearize=2),predicted_latency_us=None,
                    prediction=dict(table_known_sum_us=1250000,missing=[dict(index=1,op='relinearize'),dict(index=2,op='relinearize')]))
                (args.out/'work'/phase/'phase_result.json').write_text(json.dumps(row))
                return observation()
            from pt_agent.experiments import dacapo_compile_continuation as resource
            with patch.object(paths,'RESULTS_DIR',root),patch.object(registry,'git_rev',return_value='synthetic'),\
                    patch.object(resource,'available_ram',return_value=20*original.GIB),\
                    patch.dict(w.execute.__globals__,admission=lambda *a,**k:{},snapshot=lambda *a:None,run_process=worker):
                result=w.execute(args)
            self.assertEqual(result['status'],'ok');self.assertEqual(len(commands),6)
            paper=json.loads((args.out/'paper-record.json').read_text())
            self.assertEqual(paper['pred_latency_s'],1.25);self.assertEqual(paper['missing_price_ops'],{})
            self.assertEqual(paper['excluded_price_ops'],dict(relinearize=2))
            self.assertEqual(paper['compile_s'],.1);self.assertTrue(paper['cold'])
            self.assertIsNone(result['predicted_latency_us'])
            self.assertEqual(result['phases']['compile']['prediction']['missing'][0]['op'],'relinearize')
            self.assertTrue((args.out/'paper-prediction-policy.json').is_file())
            self.assertEqual(len((root/'runs.jsonl').read_text().splitlines()),1)

    def test_snapshot_includes_frozen_base_worker_and_new_paper_policy_sources(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);out=root/'case';out.mkdir()
            profile=root/'profile';profile.write_text('{}');template=root/'template';template.write_text('[]')
            w.snapshot(SimpleNamespace(out=out,profile=profile,metadata_template=template))
            source=out/'source'
            for name in('cold_rule63_whole_model.py','cold_rule63_worker.py','cold_rule63_paper_policy.py','a_group_profile_sum_policy.py'):
                self.assertTrue((source/name).is_file())
            self.assertEqual(json.loads((source/'paper-pricing-policy.json').read_text())['excluded_operation'],'relinearize')
            self.assertEqual(ast.dump(ast.parse((source/'cold_rule63_whole_model.py').read_text())),
                ast.dump(ast.parse(Path(original.__file__).read_text())))


if __name__=='__main__':unittest.main(verbosity=2)
