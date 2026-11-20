import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import a_group_future_closed_archive as own
from pt_agent.experiments.r34_archive_text import archive_one as original_archive_one


def fixture(root,kind,*,status='ok',raw=True):
    rows=[]
    def closed(case,run_id,record):
        case.mkdir(parents=True,exist_ok=True)
        (case/'result.json').write_text(json.dumps(record))
        (case/'completion.json').write_text(json.dumps(dict(status=record['status'],
            registry_persisted=True,required_artifacts_persisted=True)))
        rows.append(dict(run_id=run_id,round='fixture',model='fixture',mode='fixture',git_rev=None,
            metrics=dict(status=record['status']),artifacts_dir=str(case),ts='fixture'))
    parent=root/(own.A2 if kind==own.KINDS[0]else own.A5)
    source=parent if kind==own.KINDS[0]else parent/'dacapo-optimized'
    emit=source/('emit'if source==parent else 'work/emit');emit.mkdir(parents=True)
    run_id=parent.name if source==parent else parent.name+'-dacapo-optimized'
    record=dict(status=status,error=None if status=='ok'else 'fixture clean compiler error',
        run_id=run_id,compiled_bootstraps=11,compile_s=.5)
    closed(source,run_id,record)
    if source!=parent:
        closed(parent,parent.name,dict(status=status,error=record['error'],child_case=str(source),
            child_result=str(source/'result.json'),child_completion=str(source/'completion.json'),child_status=status))
    (source/'phase-resources.jsonl').write_text(json.dumps(dict(processes=[dict(pid=90001,starttime=42)]))+'\n')
    (emit/'candidate_trace.py').write_text('original public source fixture')
    (emit/'lowered.mlir').write_text('partial or complete lowered fixture')
    (emit/'lowered._hecate_golden.runtime-plan.json').write_text('{"execution":[]}')
    bundle=emit/'lowered._hecate_golden.bundle';(bundle/'data').mkdir(parents=True)
    (bundle/'manifest.json').write_text('{}');(bundle/'data/constant.bin').write_text('public constant fixture')
    trace=emit/'candidate_trace.mlir';npz=emit/'constants.npz'
    if raw:trace.write_text('public synthetic raw arithmetic text\n'*4096)
    npz.write_text('public synthetic redundant npz contents\n'*4096)
    (root/'registry.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in rows))
    args=SimpleNamespace(case=kind,out=root/'resource-archive',seconds=1140,run=True,
        no_future_consumers=True,host_pid_namespace=own.os.readlink('/proc/self/ns/pid'))
    return args,parent,source,trace,npz


class FutureClosedArchiveTests(unittest.TestCase):
    def test_exact_A2_raw_and_npz_archives_preserve_all_scientific_stats(self):
        self.assertIs(own.archive_one,original_archive_one)
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);args,parent,source,raw,npz=fixture(root,own.KINDS[0])
            before=own.snapshot(source/'result.json')
            with patch.object(own.paths,'RESULTS_DIR',root),patch.object(own,'same_birth',return_value=False), \
                 patch.object(own.resource,'setrlimit'):
                result=own.run(args)
            self.assertEqual(result['status'],'ok');self.assertEqual(result['completed_files'],2)
            self.assertEqual(result['scientific_status'],'ok');self.assertFalse(result['scientific_status_upgraded'])
            self.assertEqual(own.snapshot(source/'result.json'),before)
            for p in(raw,npz):
                self.assertFalse(p.exists());self.assertTrue(p.with_name(p.name+'.gz').is_file())
            self.assertTrue((raw.parent/'lowered._hecate_golden.bundle/data/constant.bin').is_file())
            self.assertTrue((args.out/'completion.json').is_file())

    def test_A5_clean_compiler_error_is_archived_as_independent_resource_success(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);args,parent,source,raw,npz=fixture(root,own.KINDS[1],status='error')
            before=[own.snapshot(case/name)for case in(parent,source)for name in('result.json','completion.json')]
            with patch.object(own.paths,'RESULTS_DIR',root),patch.object(own,'same_birth',return_value=False), \
                 patch.object(own.resource,'setrlimit'):
                result=own.run(args)
            self.assertEqual(result['status'],'ok');self.assertEqual(result['scientific_status'],'error')
            self.assertEqual(result['source_admission']['source_closure']['run_id'],parent.name+'-dacapo-optimized')
            self.assertEqual(result['completed_files'],2)
            self.assertEqual([own.snapshot(Path(row['path']))for row in before],before)

    def test_missing_raw_skips_adjacent_npz_and_retains_failed_scientific_status(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);args,parent,source,raw,npz=fixture(root,own.KINDS[0],status='error',raw=False)
            with patch.object(own.paths,'RESULTS_DIR',root),patch.object(own,'same_birth',return_value=False), \
                 patch.object(own.resource,'setrlimit'),patch.object(own,'archive_one')as archive:
                result=own.run(args)
            archive.assert_not_called()
            self.assertEqual(result['resource_status'],'metadata_skip')
            self.assertEqual(result['scientific_status'],'error');self.assertEqual(result['completed_files'],0)
            self.assertTrue(npz.exists());self.assertFalse(npz.with_name(npz.name+'.gz').exists())

    def test_first_admission_or_compression_error_preserves_sources_and_closes_resource_failure(self):
        for failure in('live','parent_completion','authority','namespace','compressor'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory()as temp:
                root=Path(temp);args,parent,source,raw,npz=fixture(root,own.KINDS[1])
                if failure=='parent_completion':(parent/'completion.json').unlink()
                if failure=='authority':args.no_future_consumers=False
                if failure=='namespace':args.host_pid_namespace='fixture-nonhost-namespace'
                def compressor(a,b,c):c.write_text('unfinished public fixture');raise OSError('fixture compression failed')
                with patch.object(own.paths,'RESULTS_DIR',root), \
                     patch.object(own,'same_birth',return_value=failure=='live'), \
                     patch.object(own.resource,'setrlimit'),patch.object(own,'archive_one',side_effect=compressor)as archive:
                    result=own.run(args)
                self.assertEqual(result['status'],'error');self.assertEqual(result['completed_files'],0)
                self.assertTrue(raw.exists());self.assertTrue(npz.exists())
                self.assertTrue((args.out/'completion.json').is_file())
                self.assertFalse(npz.with_name(npz.name+'.gz.partial').exists())
                self.assertEqual(archive.call_count,1 if failure=='compressor'else 0)


if __name__=='__main__':unittest.main()
