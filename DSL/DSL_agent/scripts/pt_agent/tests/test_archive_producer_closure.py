import json
import os
from pathlib import Path
import tempfile
import unittest

from pt_agent.experiments.archive_producer_closure import verify_archive_producer, ARCHIVE_MODULE, ARCHIVE_ROUND


class ArchiveProducerClosureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.archive = self.root/'archive-case'; self.owner=self.root/'serial-case'
        self.archive.mkdir(); self.owner.mkdir(); self.namespace=os.readlink('/proc/self/ns/pid')
        self.argv=['/some/python','-B','-m',ARCHIVE_MODULE,'--case','original-science','--out',str(self.archive),
                   '--host-pid-namespace',self.namespace,'--seconds','1140','--run']
        self.process=dict(exit_code=0,timed_out=False,signal=None,started_ns=100,ended_ns=500,
                          signals_sent=[],remaining_pids=[])
        self.job=dict(id='archive-job',out=str(self.archive),argv=self.argv,status='ok',experiment_status='ok',
                      cleanup_unknown=False,process=self.process)
        span=dict(event_id='20:1',pid=20,ppid=10,start_ns=200,end_ns=300,status='ok')
        self.ar=dict(round=ARCHIVE_ROUND,status='ok',resource_status='archived_or_kept',source_kind='original-science',
                     source_case=str(self.root/'original-science'),
                     source_admission={'producer_births_closed':99},
                     timing=dict(status='ok',attribution_ready=True,spans=[span]))
        self.sr=dict(round='AUTONOMOUS-SERIAL',mode='finite-trusted-serial-experiments',status='ok',jobs=[self.job])
        self.persist()
    def write(self,path,value): path.write_text(json.dumps(value))
    def persist(self):
        self.write(self.archive/'result.json',self.ar)
        self.write(self.owner/'result.json',self.sr)
        for case,result in [(self.archive,self.ar),(self.owner,self.sr)]:
            self.write(case/'completion.json',dict(round=result['round'],status=result['status'],
                required_artifacts_persisted=True,registry_persisted=True,
                phase_spans=[{'event_id':'20:1'}]if case==self.archive else []))
        self.write(self.owner/'manifest.json',{'jobs':[dict(id=self.job['id'],out=str(self.archive),argv=self.argv)]})
        self.write(self.owner/'archive-job-command.json',self.argv)
        self.write(self.owner/'supervisor.json',{'pid':10,'process':{'pid':10,'starttime':1000}})
        self.resource={'ts_ns':150,'outer_process':{'pid':10,'starttime':1000},
                       'processes':[{'pid':20,'starttime':2000,'ppid':10,'pgid':20}]}
        (self.owner/'archive-job-resources.jsonl').write_text(json.dumps(self.resource)+'\n')
        events=[{'event_id':'20:1','pid':20,'type':kind}for kind in ('begin','end')]
        (self.archive/'events.jsonl').write_text(''.join(json.dumps(e)+'\n'for e in events))
        self.registry=[dict(run_id=case.name,round=result['round'],model=model,mode=mode,git_rev=None,
            metrics={'status':result['status'],**({'jobs':[self.job]}if case==self.owner else{})},
            artifacts_dir=str(case),ts='2026-11-17T00:00:00+00:00')for case,result,model,mode in [
            (self.archive,self.ar,'original-science','resource-only-reversible-gzip'),
            (self.owner,self.sr,'original-model-experiment-queue','finite-trusted-serial-experiments')]]
        (self.root/'registry.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in self.registry))
    def verify(self,reader=lambda pid:None):
        return verify_archive_producer(self.archive,self.owner,host_pid_namespace=self.namespace,process_reader=reader)
    def test_actual_job_birth_and_event_binding_without_archive_own_resources(self):
        proof=self.verify()
        self.assertEqual(proof['closed_producer_births'],1)
        self.assertEqual(proof['archive_producer_births'],[dict(pid=20,starttime=2000,host_pid_reused=False)])
        self.assertEqual(proof['archive_event_pids'],[20])
        self.assertFalse(proof['source_scientific_births_used_as_archive_proof'])
        self.assertEqual(proof['owning_controller_birth']['starttime'],1000)
        self.assertEqual(proof['exact_archive_argv'],self.argv)
    def test_source_birth_count_and_wrong_or_missing_job_snapshots_do_not_substitute(self):
        path=self.owner/'archive-job-resources.jsonl'; original=path.read_text()
        mutations=[[],[dict(self.resource,processes=[])],
            [dict(self.resource,processes=[{'pid':30,'starttime':3000,'ppid':10,'pgid':30}])],
            [dict(self.resource,processes=[{'pid':20,'starttime':0,'ppid':10,'pgid':20}])],
            [dict(self.resource,processes=[{'pid':20,'starttime':True,'ppid':10,'pgid':20}])],
            [dict(self.resource,outer_process={'pid':10,'starttime':999})],
            [dict(self.resource,ts_ns=501)]]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                path.write_text(''.join(json.dumps(x)+'\n'for x in mutation))
                with self.assertRaises(ValueError):self.verify()
        path.write_text(original)
    def test_live_original_birth_including_zombie_rejected_pid_reuse_allowed(self):
        for live in [dict(pid=20,starttime=2000,state='R'),dict(pid=20,starttime=2000,state='Z')]:
            with self.subTest(live=live),self.assertRaises(ValueError):self.verify(lambda pid:live if pid==20 else None)
        with self.assertRaises(ValueError):self.verify(lambda pid:dict(pid=10,starttime=1000)if pid==10 else None)
        proof=self.verify(lambda pid:dict(pid=pid,starttime=9000,state='R'))
        self.assertTrue(proof['archive_producer_births'][0]['host_pid_reused'])
    def test_exact_manifest_command_registry_and_cleanup_required(self):
        command=self.owner/'archive-job-command.json';self.write(command,self.argv+['--unexpected'])
        with self.assertRaises(ValueError):self.verify()
        self.persist();self.registry.append(self.registry[0])
        (self.root/'registry.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in self.registry))
        with self.assertRaises(ValueError):self.verify()
        self.persist();self.job['cleanup_unknown']=True;self.persist()
        with self.assertRaises(ValueError):self.verify()
    def test_actual_event_pid_and_complete_event_pairs_required(self):
        event_path=self.archive/'events.jsonl'; original=event_path.read_text()
        for events in [[{'event_id':'30:1','pid':30,'type':k}for k in ('begin','end')],
                       [{'event_id':'20:1','pid':20,'type':'begin'}]]:
            with self.subTest(events=events):
                event_path.write_text(''.join(json.dumps(x)+'\n'for x in events))
                with self.assertRaises(ValueError):self.verify()
        event_path.write_text(original)
    def test_closed_parent_observations_keep_unrelated_failure_status(self):
        self.sr['status']='completed_with_observations';self.persist()
        self.assertEqual(self.verify()['closed_producer_births'],1)


    def test_symlinked_resource_metadata_and_wrong_namespace_rejected(self):
        resource=self.owner/'archive-job-resources.jsonl'
        original=self.owner/'saved-metadata.jsonl';resource.rename(original);resource.symlink_to(original)
        with self.assertRaises(ValueError):self.verify()
        resource.unlink();original.rename(resource)
        with self.assertRaises(ValueError):
            verify_archive_producer(self.archive,self.owner,host_pid_namespace='pid:[wrong]',process_reader=lambda pid:None)

if __name__=='__main__':unittest.main()
