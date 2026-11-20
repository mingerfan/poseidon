"""Four new synthetic archive checks; no original case, payload readback or hash."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import a_group_closed_raw_archive as m
from pt_agent.experiments.r34_archive_text import archive_one as original_archive_one


def fixture(root):
    ledger=[]
    def closed(case,result):
        case.mkdir(parents=True,exist_ok=True)
        (case/'result.json').write_text(json.dumps(result))
        (case/'completion.json').write_text(json.dumps(dict(status=result['status'],registry_persisted=True,required_artifacts_persisted=True)))
        ledger.append(dict(run_id=case.name,round='fixture',model=result.get('model','fixture'),mode='fixture',git_rev=None,metrics=dict(status=result['status']),artifacts_dir=str(case),ts='fixture'))
    queue=root/m.QUEUE;closed(queue,dict(status='completed_with_observations'))
    (queue/'supervisor.json').write_text(json.dumps(dict(pid=231062,process=dict(starttime=5241634))))
    jobs=[dict(out=str(root/name),argv=[])for name in m.CASES]+[dict(out='unused-a3',argv=[]),dict(out='unused-a4',argv=[])]
    (queue/'manifest.json').write_text(json.dumps(dict(jobs=jobs)))
    wait=root/'a4-resnet20-123-prepared-disk-keyfree-as4-r31-20261117';closed(wait,dict(status='error'))
    (wait/'fixture-resources.jsonl').write_text(json.dumps(dict(processes=[dict(pid=90001,starttime=100)]))+'\n')
    raw=[]
    for index,(name,model)in enumerate(zip(m.CASES,m.MODELS)):
        case=root/name;emit=case/'emit';emit.mkdir(parents=True);(case/'plans').mkdir()
        plan=emit/'lowered._hecate_golden.runtime-plan.json';plan.write_text('{}')
        bundle=emit/'lowered._hecate_golden.bundle';(bundle/'data').mkdir(parents=True);(bundle/'manifest.json').write_text('{}')
        phases=[dict(phase=p,worker=dict(runtime_plan=str(plan))if p=='observe'else{})for p in('planning','emit','trace','compile','observe')]
        count=dict.fromkeys(('planning','emit','trace','compile','observe'),1)
        result=dict(status='ok',round='A2-NATIVE-PROFILE',model=model,monitor_stopped=True,error=None,phases=phases,attempted=count,completed=count,compiled_bootstraps=index+1,compile_s=.5)
        closed(case,result)
        for n in('events.jsonl','program.ptdsl','source.json','admission.json','compile-argv.json','placement.json'):(case/n).write_text('fixture')
        for n in('depth-dp','depth-dp+prune','stage-cut','stage-cut+prune'):(case/'plans'/(n+'.json')).write_text('{}')
        for n in('candidate_trace.py','constants.npz','lowered.mlir'):(emit/n).write_text('preserved fixture')
        (case/'fixture-resources.jsonl').write_text(json.dumps(dict(processes=[dict(pid=90002+index,starttime=101+index)]))+'\n')
        source=emit/'candidate_trace.mlir';source.write_text('public synthetic text\n'*1024);raw.append(source)
    (root/'registry.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in ledger))
    manifest=dict(round=m.ROUND,no_future_A2_raw_consumers=True,host_pid_namespace=m.os.readlink('/proc/self/ns/pid'),wait_closed_case=dict(path=str(wait),owner_pid=236794,owner_starttime=5280683),files=[m.file_stat(p)for p in raw])
    path=root/'archive-manifest.json';path.write_text(json.dumps(manifest))
    return raw,manifest,SimpleNamespace(run=True,seconds=1200,out=root/'archive',manifest=path)


class ArchiveCases(unittest.TestCase):
    def test_exact_three_real_tiny_archives_keep_scientific_files_and_complete_record(self):
        self.assertIs(m.archive_one,original_archive_one)
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);raw,manifest,args=fixture(root)
            with patch.object(m.paths,'RESULTS_DIR',root),patch.object(m,'same_birth',return_value=False),patch.object(m.resource,'setrlimit'):
                result=m.run(args)
            self.assertEqual(result['status'],'ok');self.assertEqual(result['completed_files'],3);self.assertGreater(result['released_bytes'],0)
            for source in raw:
                self.assertFalse(source.exists());self.assertTrue(source.with_name(source.name+'.gz').is_file());self.assertFalse(source.with_name(source.name+'.gz.partial').exists())
                self.assertTrue((source.parent/'constants.npz').is_file());self.assertTrue((source.parent/'lowered._hecate_golden.runtime-plan.json').is_file())
            rows=[json.loads(line)for line in(root/'registry.jsonl').read_text().splitlines()if json.loads(line)['run_id']=='archive']
            self.assertEqual(len(rows),1);self.assertEqual(set(rows[0]),m.FIELDS)
            completion=json.loads((args.out/'completion.json').read_text());self.assertEqual(completion['status'],'ok');self.assertTrue(completion['registry_persisted'])

    def test_failure_preserves_raw_partial_and_stops_before_next_source(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);raw,manifest,args=fixture(root)
            def fail(source,target,temporary):temporary.write_text('unfinished public fixture');raise OSError('fixture compressor failed')
            with patch.object(m.paths,'RESULTS_DIR',root),patch.object(m,'same_birth',return_value=False),patch.object(m.resource,'setrlimit'),patch.object(m,'archive_one',side_effect=fail):
                result=m.run(args)
            self.assertEqual(result['status'],'error');self.assertEqual(result['completed_files'],0);self.assertIn('fixture compressor failed',result['error'])
            self.assertTrue(all(p.exists()for p in raw));self.assertTrue(raw[0].with_name(raw[0].name+'.gz.partial').is_file());self.assertFalse(raw[1].with_name(raw[1].name+'.gz.partial').exists())
            self.assertTrue(result['interrupted_state']['source_exists']);self.assertTrue(result['interrupted_state']['temporary_exists'])
            self.assertEqual(json.loads((args.out/'completion.json').read_text())['status'],'error')

    def test_resource_setup_error_retains_terminal_metadata_without_compression(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);raw,manifest,args=fixture(root)
            with patch.object(m.paths,'RESULTS_DIR',root),patch.object(m.resource,'setrlimit',side_effect=OSError('fixture AS setup failed')):
                result=m.run(args)
            self.assertEqual(result['status'],'error');self.assertIn('fixture AS setup failed',result['error'])
            self.assertEqual(result['completed_files'],0);self.assertTrue(all(p.exists()for p in raw))
            self.assertEqual(json.loads((args.out/'completion.json').read_text())['status'],'error')

    def test_live_queue_or_A4_owner_rejects_admission_before_compression(self):
        for active in(231062,236794):
            with self.subTest(active=active),tempfile.TemporaryDirectory()as tmp:
                root=Path(tmp);raw,manifest,args=fixture(root)
                with patch.object(m.paths,'RESULTS_DIR',root),patch.object(m,'same_birth',side_effect=lambda pid,birth:pid==active):
                    with self.assertRaises(ValueError):m.validate_manifest(manifest)
                self.assertTrue(all(p.is_file()for p in raw))

    def test_dangling_archive_output_is_rejected_without_mutation(self):
        for suffix in('.gz','.gz.partial'):
            with self.subTest(suffix=suffix),tempfile.TemporaryDirectory()as tmp:
                root=Path(tmp);raw,manifest,args=fixture(root)
                link=raw[0].with_name(raw[0].name+suffix);link.symlink_to(root/'missing-target')
                with patch.object(m.paths,'RESULTS_DIR',root),patch.object(m,'same_birth',return_value=False):
                    with self.assertRaisesRegex(ValueError,'Archive output must be fresh'):m.validate_manifest(manifest)
                self.assertTrue(link.is_symlink());self.assertTrue(all(p.is_file()for p in raw))

    def test_file_stat_rejects_symbolic_and_hard_links(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);p=root/'raw.mlir';p.write_text('public fixture');link=root/'linked.mlir';link.symlink_to(p)
            with self.assertRaises(ValueError):m.file_stat(link)
            hard=root/'hard.mlir';hard.hardlink_to(p)
            with self.assertRaises(ValueError):m.file_stat(p)


if __name__=='__main__':unittest.main(verbosity=2)
