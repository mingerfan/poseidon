"""Synthetic closure/lease/archive metadata; original scientific payloads unused."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import closed_success_raw_trace_archive as archive
from pt_agent.experiments import r34_archive_text


class ResourceFixture(unittest.TestCase):
    def test_closed_sources_original_primitive_global_lease_and_strict_metadata(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);case=root/'synthetic-producer';emit=case/'emit';emit.mkdir(parents=True)
            source=emit/'candidate_trace.mlir';source.write_text('synthetic compiler text\n'*4000)
            plan=emit/'lowered.runtime-plan.json';plan.write_text('{}\n')
            bundle=emit/'lowered.bundle';(bundle/'data').mkdir(parents=True)
            (bundle/'manifest.json').write_text('{}\n');(bundle/'data/constant').write_text('retained fixture\n')
            (case/'weights.npz').write_text('synthetic retained file, not a numeric model\n')
            clean=dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])
            result=dict(round='SYNTHETIC',model='tiny',mode='fixture',status='ok',error=None,
                attempted=dict(trace=1,compile=1),completed=dict(trace=1,compile=1),
                processes=[dict(phase='compile',observation=clean)],compiled_bootstraps=1)
            completion=dict(status='ok',registry_persisted=True,required_artifacts_persisted=True)
            (case/'result.json').write_text(json.dumps(result));(case/'completion.json').write_text(json.dumps(completion))
            resources=case/'resources.jsonl'
            resources.write_text(json.dumps(dict(processes=[dict(pid=123,starttime=456)]))+'\n')
            registry=dict(run_id=case.name,round='SYNTHETIC',model='tiny',mode='fixture',git_rev=None,
                metrics=dict(status='ok'),artifacts_dir=str(case),ts='fixture')
            (root/'registry.jsonl').write_text(json.dumps(registry)+'\n')
            manifest=dict(schema='closed-success-raw-trace-archive-v1',no_future_raw_consumers=True,
                host_pid_namespace=os.readlink('/proc/self/ns/pid'),files=[dict(case=str(case),source=str(source),
                    stat=archive.snapshot(source,archive_source=True),expected_round='SYNTHETIC',
                    expected_model='tiny',compiled_bootstraps=1)])
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            with patch.object(archive.paths,'RESULTS_DIR',root),patch.object(archive.metadata,'same_birth',return_value=False):
                self.assertIs(archive.archive_one,r34_archive_text.archive_one)
                selected=archive.source_gate(manifest)
                self.assertEqual(selected[0]['producer_births_closed'],1)
                bad=dict(manifest,host_pid_namespace='pid:[wrong]')
                with self.assertRaisesRegex(ValueError,'host process namespace'):archive.source_gate(bad)
                with patch.object(archive.metadata,'same_birth',return_value=True):
                    with self.assertRaisesRegex(ValueError,'producer births must be closed'):archive.source_gate(manifest)
                resources.write_text(json.dumps(dict(processes=[dict(pid=123,starttime=0)]))+'\n')
                with self.assertRaisesRegex(ValueError,'positive PID/starttime'):archive.source_gate(manifest)
                resources.write_text(json.dumps(dict(processes=[dict(pid=123,starttime=456)]))+'\n')
                target=source.with_name(source.name+'.gz');target.symlink_to(root/'missing')
                with self.assertRaisesRegex(ValueError,'existing archives'):archive.source_gate(manifest)
                target.unlink()
                observed=archive.run(SimpleNamespace(out=root/'archive-resource',manifest=path,seconds=30,run=True))
                self.assertEqual(observed['status'],'ok');self.assertEqual(observed['completed_files'],1)
                self.assertTrue(observed['heavy_lease']['original_shared_global_lock'])
                self.assertTrue(observed['heavy_lease']['released'])
                self.assertFalse(source.exists());self.assertTrue(target.is_file())
                self.assertFalse(target.with_name(target.name+'.partial').exists())
                self.assertEqual(json.loads((case/'result.json').read_text()),result)
                self.assertEqual(json.loads((case/'completion.json').read_text()),completion)
                self.assertTrue(plan.is_file());self.assertTrue((bundle/'manifest.json').is_file())
                self.assertTrue((case/'weights.npz').is_file())
                self.assertEqual(observed['model_runs'],observed['native_compiles'])
                self.assertEqual(observed['API_calls'],0);self.assertEqual(observed['integrity_or_readback_checks'],0)
                saved=json.loads((root/'archive-resource/completion.json').read_text())
                self.assertEqual(saved['status'],'ok');self.assertTrue(saved['registry_persisted'])
                rows=[json.loads(line)for line in (root/'registry.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),2);self.assertEqual(rows[0],registry)
                self.assertEqual(set(rows[1]),set(registry))


if __name__=='__main__':unittest.main()
