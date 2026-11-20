"""Tiny closed metadata and fake-R34 fixtures; no scientific archive run."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.experiments import bert_rule63_restored_input_archive as own
from pt_agent.experiments import r34_archive_text


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class RecoveredInputArchive(unittest.TestCase):
    def fixture(self,root,*,status='error',raw=True):
        case=root/own.SOURCE_ID
        result=dict(round=own.producer.ROUND,mode=own.producer.MODE,model=own.producer.MODEL,
            run_id=case.name,status=status,error='synthetic closed compiler failure'if status=='error'else None,
            monitor_stopped=True,remaining_pids=[],cleanup_unknown=False)
        write(case/'result.json',result);write(case/'completion.json',dict(round=result['round'],status=status,
            registry_persisted=True,required_artifacts_persisted=True))
        write(case/'input/plan/manifest.json',dict(synthetic_retained_plan=True))
        write(case/'input/constants-bundle/manifest.json',dict(synthetic_retained_bundle=True))
        if raw:(case/'input/candidate_trace.mlir').write_text('tiny recovered input fixture')
        original=root/own.producer.SOURCE_ID/'work/emit/candidate_trace.mlir.gz'
        original.parent.mkdir(parents=True);original.write_text('retained original gzip fixture; not decoded')
        ledger=dict(run_id=case.name,round=result['round'],model=result['model'],mode=result['mode'],git_rev='fixture',
            metrics=dict(status=status),artifacts_dir=str(case),ts='fixture')
        (root/'registry.jsonl').write_text(json.dumps(ledger)+'\n')
        args=SimpleNamespace(out=root/'resource-job',run=True,seconds=100,
            host_pid_namespace=os.readlink('/proc/self/ns/pid'),no_future_restored_input_consumers=True)
        return args,case,result

    def test_exact_closed_success_or_error_selects_one_copy_missing_skips_unknown_cleanup_rejects(self):
        for status in('ok','error'):
            with self.subTest(status=status),tempfile.TemporaryDirectory()as temporary,\
                    patch.object(paths,'RESULTS_DIR',Path(temporary)),patch.object(own.prior,'producer_births',return_value=3):
                args,case,result=self.fixture(Path(temporary),status=status)
                proof=own.admit(args)
                self.assertEqual(proof['scientific_status'],status)
                self.assertEqual(len(proof['selected']),1)
                self.assertEqual(Path(proof['selected'][0]['source']),case/'input/candidate_trace.mlir')
                result['cleanup_unknown']=True;write(case/'result.json',result)
                with self.assertRaisesRegex(ValueError,'unknown'):own.admit(args)
        with tempfile.TemporaryDirectory()as temporary,patch.object(paths,'RESULTS_DIR',Path(temporary)),\
                patch.object(own.prior,'producer_births',return_value=0):
            args,_,_=self.fixture(Path(temporary),raw=False)
            proof=own.admit(args);self.assertFalse(proof['selected']);self.assertIsNotNone(proof['skip_reason'])

    def fake_run(self,root,*,primitive_failed=False):
        args,case,_=self.fixture(root,status='error')
        calls=[]
        def primitive(source,target,temporary):
            calls.append(str(source));temporary.write_text('synthetic temporary')
            if primitive_failed:raise RuntimeError('synthetic compression failure')
            count=source.stat().st_size;temporary.rename(target);source.unlink()
            return dict(source=str(source),archive=str(target),status='compressed',
                raw_bytes=count,compressed_bytes=19,released_bytes=count-19)
        with patch.object(own.prior,'producer_births',return_value=3),\
                patch.object(own,'archive_one',side_effect=primitive),\
                patch.object(own.resource,'setrlimit')as limits,\
                patch.object(own.shutil,'disk_usage',return_value=SimpleNamespace(free=100*own.GIB)):
            result=own.run(args)
        self.assertEqual([c.args[1]for c in limits.call_args_list],[(own.GIB,own.GIB),(16*own.GIB,16*own.GIB)])
        return args,case,result,calls

    def test_fake_primitive_success_closes_resource_case_without_upgrading_science(self):
        self.assertIs(own.archive_one,r34_archive_text.archive_one)
        with tempfile.TemporaryDirectory()as temporary,patch.object(paths,'RESULTS_DIR',Path(temporary)):
            root=Path(temporary);args,case,result,calls=self.fake_run(root)
            self.assertEqual(len(calls),1);self.assertEqual(result['status'],'ok')
            self.assertEqual(result['scientific_status'],'error');self.assertFalse(result['scientific_status_upgraded'])
            self.assertEqual(own.prior.read(case/'result.json')['status'],'error')
            self.assertTrue((case/'input/candidate_trace.mlir.gz').exists())
            self.assertTrue((case/'input/plan/manifest.json').exists())
            self.assertTrue((case/'input/constants-bundle/manifest.json').exists())
            self.assertTrue((root/own.producer.SOURCE_ID/'work/emit/candidate_trace.mlir.gz').exists())
            self.assertEqual(own.prior.read(args.out/'completion.json')['status'],'ok')
            rows=[json.loads(line)for line in(root/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2);self.assertTrue(all(len(row)==8 for row in rows))
            self.assertEqual(rows[-1]['metrics']['scientific_status'],'error')

    def test_fake_primitive_failure_retains_raw_partial_and_terminal_resource_failure(self):
        with tempfile.TemporaryDirectory()as temporary,patch.object(paths,'RESULTS_DIR',Path(temporary)):
            args,case,result,calls=self.fake_run(Path(temporary),primitive_failed=True)
            self.assertEqual(len(calls),1);self.assertEqual(result['status'],'error')
            self.assertEqual(result['scientific_status'],'error')
            self.assertTrue((case/'input/candidate_trace.mlir').exists())
            self.assertTrue((case/'input/candidate_trace.mlir.gz.partial').exists())
            self.assertEqual(own.prior.read(args.out/'completion.json')['status'],'error')
            self.assertTrue(result['monitor_stopped'])


if __name__=='__main__':unittest.main()
