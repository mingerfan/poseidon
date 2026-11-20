"""Only new wrapper/consumer metadata fixtures; no real gates or archive."""
import copy
import json
from pathlib import Path
import tempfile
from types import FunctionType,SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_inventory32_closed_trace_archive as entry


def manifest(args):
    jobs=[]
    for index,(module,case) in enumerate((
        ('pt_agent.experiments.qwen_retained_component_replay',args.source_case),
        ('pt_agent.experiments.qwen_layer_native_probe',args.probe_case),
        ('pt_agent.experiments.qwen_remaining_prefill_compile',args.remaining_case),
        ('pt_agent.experiments.qwen_generation_kernel_compile',args.kernel_case),
        ('pt_agent.experiments.qwen_component_link_validation',args.results_root/(entry.PREFIX+'original27-typed-link-validation-20261117')))):
        argv=['python3','-B','-m',module,'--out',str(case)]
        if index:argv+=['--source-case',str(args.source_case)]
        else:argv+=['--completed-cost-from',str(args.results_root/'historical-cost-authority'),
                    '--logical-constant-inventory-gib','32']
        jobs.append(dict(id=str(index),out=str(case),argv=argv))
    return dict(deadline_utc=entry.base.DEADLINE.isoformat(),jobs=jobs)


class InventoryObserverChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.patch=patch.object(entry.base.paths,'RESULTS_DIR',self.root);self.patch.start();self.addCleanup(self.patch.stop)
        self.ns=entry._namespace();self.args=self.ns['parser']().parse_args(['--out',str(self.root/'observer'),'--run'])

    def test_defaults_and_new_fixed_family_accept_original_family_rejected(self):
        self.assertEqual(self.args.watch_pid,219446);self.assertEqual(self.args.watch_starttime,5038402)
        self.assertEqual(self.ns['validate_args'](self.args),self.root)
        for name,suffix in entry.CASE_NAMES.items():self.assertEqual(getattr(self.args,name+'_case'),self.root/suffix)
        old=entry.base.parser().parse_args(['--out',str(self.root/'old-observer'),'--run'])
        with self.assertRaises(ValueError):self.ns['validate_args'](old)
        with self.assertRaises(ValueError):entry.base.validate_args(self.args)
        self.args.watch_starttime+=1
        with self.assertRaises(ValueError):self.ns['validate_args'](self.args)

    def test_all_original_own_functions_clone_exact_code_and_isolate_globals(self):
        second=entry._namespace();allowed={'MODULE','ROUND','MODE','WATCH_PID','WATCH_START','PREFIX','CASE_NAMES'}
        for name,value in vars(entry.base).items():
            if isinstance(value,FunctionType) and value.__globals__ is vars(entry.base):
                self.assertIs(self.ns[name].__code__,value.__code__)
                self.assertIs(self.ns[name].__globals__,self.ns)
                self.assertIs(second[name].__globals__,second)
                self.assertIs(self.ns[name].__defaults__,value.__defaults__)
            elif name not in allowed:self.assertIs(self.ns[name],value)
        self.ns['CASE_NAMES']['source']='private mutation'
        self.assertNotEqual(self.ns['CASE_NAMES'],second['CASE_NAMES'])
        self.assertEqual(second['CASE_NAMES'],entry.CASE_NAMES)
        self.assertEqual(entry.base.WATCH_PID,199735);self.assertEqual(entry.base.WATCH_START,4311840)
        self.assertEqual(entry.base.PREFIX,'qwen-retained13-component-expanded-wall-')
        self.assertEqual(second['TRACE_NAMES'],('candidate_trace.mlir','candidate_trace.segment.earth.mlir'))
        self.assertIs(second['DEADLINE'],entry.base.DEADLINE)
        self.assertIs(second['archive_one'],entry.base.archive_one)

    def test_original_five_job_consumer_proof_for_inventory32_and_rejections(self):
        self.args.watch_case.mkdir();path=self.args.watch_case/'manifest.json';value=manifest(self.args)
        path.write_text(json.dumps(value));rows=self.ns['consumer_proof'](self.args)
        self.assertEqual(len(rows),5)
        original=copy.deepcopy(value)
        value['jobs'][2]['argv']+=['--input','candidate_trace.mlir'];path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):self.ns['consumer_proof'](self.args)
        value=copy.deepcopy(original);value['jobs'].pop();path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):self.ns['consumer_proof'](self.args)
        path.write_text(json.dumps(original))
        extra=self.root/'pending.json';extra.write_text(json.dumps(dict(jobs=[dict(argv=['python3',str(self.args.probe_case)])])))
        self.args.consumer_manifest=[extra]
        with self.assertRaises(ValueError):self.ns['consumer_proof'](self.args)

    def test_pinned_supervisor_and_unique_future_heavy_owner_metadata_only(self):
        self.args.watch_case.mkdir()
        (self.args.watch_case/'supervisor.json').write_text(json.dumps(dict(pid=entry.WATCH_PID,process=dict(starttime=entry.WATCH_START))))
        self.ns['same_live']=lambda pid,birth:(pid,birth)==(219446,5038402)
        self.ns['lease_owners']=lambda root:[219446]
        proof=self.ns['watch_proof'](self.args)
        self.assertTrue(proof['live']);self.assertEqual(proof['lease_owners'],[219446])
        self.ns['lease_owners']=lambda root:[199735]
        with self.assertRaises(ValueError):self.ns['watch_proof'](self.args)
        (self.args.watch_case/'supervisor.json').write_text(json.dumps(dict(pid=219446,process=dict(starttime=1))))
        with self.assertRaises(ValueError):self.ns['watch_proof'](self.args)

    def test_owned_worker_argv_uses_wrapper_without_any_child_launch(self):
        case=self.args.probe_case;emit=case/'codegen/emit';emit.mkdir(parents=True)
        source=emit/'candidate_trace.mlir';source.write_text('public text fixture')
        self.args.out.mkdir();component=dict(name='layer0',case=str(case))
        proof=self.ns['file_proof'](component,source.name)
        self.ns['disk_guard']=lambda *a,**k:{}
        observed=[]
        class Child:
            pid=123456;returncode=0
            def poll(self):return 0
        def child(argv,**kwargs):observed.append(argv);return Child()
        self.ns['process']=lambda pid:dict(starttime=10)
        self.ns['stop_owned']=lambda *a:[]
        with patch.object(entry.base.subprocess,'Popen',side_effect=child):
            result=self.ns['run_archive'](self.args,component,source.name,__import__('time').monotonic()+5)
        self.assertEqual(observed[0][observed[0].index('-m')+1],entry.MODULE)
        request=entry.base.read(Path(observed[0][observed[0].index('--archive-request')+1]))
        self.assertEqual(request['args']['watch_pid'],219446)
        self.assertEqual(request['args']['watch_starttime'],5038402)
        self.assertEqual(request['file'],proof)
        self.assertEqual(result['status'],'error')  # No fake success response was manufactured.

    def test_main_dispatches_owned_worker_through_same_private_namespace(self):
        self.ns['archive_worker']=lambda request,response:17
        with patch.object(entry,'_namespace',return_value=self.ns):
            value=entry.main(['--archive-request',str(self.root/'own.request.json'),
                              '--archive-response',str(self.root/'own.response.json')])
        self.assertEqual(value,17)
        self.assertNotIn('archive_worker',vars(entry))


if __name__=='__main__':unittest.main()
