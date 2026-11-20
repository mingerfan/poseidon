"""Small metadata and fake-process checks; no model, compiler, samples or API."""
from argparse import Namespace
from contextlib import ExitStack, redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_vgg_cpu_overlap as entry


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+'\n' if isinstance(value, dict) else value)


def clean(**changes):
    result = dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])
    result.update(changes)
    return result


class VGGOverlapChecks(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.args = Namespace(out=self.root/'heir-vgg16-fused-pool-conv-cpu-overlap-fixture',
            registry=self.root/'registry.jsonl', run=True, memory_gib=24,
            python=Path('/fixture/unchanged-intake-python'), heir_opt=self.root/'heir-opt',
            bert_case=self.root/'bert-source', bert_run_case=self.root/'bert-run', batch_case=self.root/'batch',
            bert_pid=64170, bert_starttime=1157046, batch_pid=36762, batch_starttime=961036,
            **{name: None for name in entry.original.SOURCE_FIELDS})
        put(self.args.heir_opt, 'fake compiler executable metadata only\n'); self.args.heir_opt.chmod(0o700)
        put(self.args.bert_case/'session/SESSION.json', dict(phase='tune', iteration=3, calls=3, failed=None))
        put(self.args.bert_run_case/'result.json', dict(round='BERT-SERIAL-CONTINUATION', model='bert-tiny-sst2',
            status='running', termination=None, source=dict(source_case=str(self.args.bert_case)),
            attempted=dict(step=1), completed=dict(step=0)))
        put(self.args.batch_case/'supervisor.json', dict(pid=self.args.batch_pid,
            process=dict(pid=self.args.batch_pid, starttime=self.args.batch_starttime)))
        put(self.args.batch_case/'active.json', dict(status='running', id='original-bert', out=str(self.args.bert_run_case),
            argv=['python3', '-B', '-m', 'pt_agent.experiments.bert_serial_continuation',
                  '--source-case', str(self.args.bert_case)]))
        self.source = {name: self.root/('original-'+name) for name in entry.original.SOURCE_FIELDS}
        put(self.source['native_parameters'], dict(logN=16, Q=[101], P=[103], logDefaultScale=45))
        self.checked = dict(contract={'role':'fake-original'}, native_parameters=dict(logN=16,Q=[101],P=[103],logDefaultScale=45),
            paths={name:str(path) for name,path in self.source.items()},
            pipeline='builtin.module(test{parameter-file='+str(self.source['native_parameters'])+'})')
        self.table = self.process_table(); self.names = {}; self.calls = []
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(entry.paths, 'RESULTS_DIR', self.root))
        self.stack.enter_context(patch.object(entry.overlap, '_scan', side_effect=lambda:self.table))
        self.stack.enter_context(patch.object(entry.overlap, '_lease_owners', return_value=[self.args.batch_pid]))
        self.stack.enter_context(patch.object(entry.overlap, '_process_name', side_effect=lambda pid:self.names.get(pid,'python3')))
        self.stack.enter_context(patch.object(entry.original, '_available_ram_bytes', return_value=12*entry.GIB))
        self.stack.enter_context(patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=40*entry.GIB)))

    def process_table(self):
        return {36762:dict(pid=36762,ppid=1,starttime=961036,state='S'),
            40000:dict(pid=40000,ppid=36762,starttime=5,state='S'),
            64170:dict(pid=64170,ppid=40000,starttime=1157046,state='R'),
            os.getpid():dict(pid=os.getpid(),ppid=1,starttime=9,state='R',vmrss_kib=2048)}

    def observe(self, sample=None, available=None, free=None):
        return entry.parallel_observation(self.args, sample,
            12*entry.GIB if available is None else available, 40*entry.GIB if free is None else free)

    def fake(self, *, bad=None, after=None, native_check=None, process=None, source_failure=False):
        from pt_agent.experiments import heir_common_model_pipeline
        def build(command, **options):
            self.calls.append((command, options))
            put(Path(options['log']), 'synthetic process observation only\n')
            put(Path(options['resources']), '')
            if process is not None:
                return process(command, options)
            if '--worker' in command:
                out = Path(command[command.index('--out')+1])
                put(out/'result.json',dict(status='ok',lowered_adapter_contract={'role':'fake-lowered'}))
                put(out/'model.mlir','synthetic source metadata only\n')
            if after is not None:
                after(len(self.calls))
            return clean(**(bad or {}))
        def snapshot(out):
            (out/'source').mkdir(); return []
        with ExitStack() as stack:
            stack.enter_context(patch.object(heir_common_model_pipeline, 'sources', return_value=self.source))
            gate = stack.enter_context(patch.object(entry.original, 'source_gate',
                side_effect=ValueError('synthetic source rejected') if source_failure else None,
                return_value=self.checked))
            stack.enter_context(patch.object(entry.original, '_snapshot', side_effect=snapshot))
            stack.enter_context(patch.object(entry.original, 'run_process', side_effect=build))
            check = stack.enter_context(patch.object(entry.original, 'check_worker', return_value={'passed':True}))
            stack.enter_context(patch.object(entry.original, 'check_native',
                side_effect=native_check, return_value={'static_bootstrap_operations':2}))
            result = entry.execute(self.args)
        self.gate_call = gate.call_args; self.worker_check_call = check.call_args
        return result

    def test_original_two_phase_commands_parameters_and_resources_are_preserved(self):
        result = self.fake()
        self.assertEqual(result['status'],'ok'); self.assertEqual(len(self.calls),2)
        worker, native = self.calls[0][0], self.calls[1][0]
        self.assertIn(entry.original.MODULE,worker); self.assertNotIn(entry.MODULE,worker)
        self.assertEqual(worker[worker.index('--memory-gib')+1],'24')
        self.assertEqual(worker[worker.index('--seconds')+1],'1860')
        self.assertEqual(worker[worker.index('--python')+1],str(self.args.python))
        for name,path in self.source.items():
            self.assertEqual(worker[worker.index('--'+name.replace('_','-'))+1],str(path))
        self.assertEqual(native[0],str(self.args.heir_opt)); self.assertIn('--mlir-disable-threading',native)
        self.assertIn('--pass-pipeline=builtin.module(test{parameter-file='+str(self.args.out/'native/native-parameters.json')+'})',native)
        self.assertEqual(native[native.index('-o')+1],str(self.args.out/'native/model-lattigo.mlir'))
        self.assertEqual(entry.original.read(self.args.out/'native/native-parameters.json'),self.checked['native_parameters'])
        self.assertEqual([options['mem_gib'] for _,options in self.calls],[24,24])
        self.assertEqual([options['fsize_gib'] for _,options in self.calls],[16,16])
        self.assertLessEqual(self.calls[0][1]['timeout'],1920);self.assertLessEqual(self.calls[1][1]['timeout'],7200)
        self.assertEqual(self.worker_check_call.args[1],self.checked)
        self.assertEqual(self.gate_call.args[0],self.source)

    def test_success_retains_fact_acceptance_registry_completion_and_scope(self):
        result = self.fake()
        self.assertTrue(result['model_compiled']);self.assertTrue(result['model_compiled_accepted'])
        self.assertTrue(result['strict_math_passed'])
        self.assertEqual(result['completed'],dict(export_math=1,native_compile=1))
        self.assertEqual(result['static_native_bootstraps'],2)
        self.assertEqual(result['limits'],dict(total_seconds=9600,worker_seconds=1800,nix_seconds=1860,
            worker_outer_seconds=1920,native_seconds=7200,AS_gib=24,subtree_RSS_gib=10,
            minimum_available_gib=2,case_gib=32,file_gib=16,minimum_free_disk_gib=10,concurrency=1,
            monitor_poll_seconds=.2))
        self.assertFalse(result['global_heavy_lease_acquired_by_this_entry']);self.assertEqual(result['bert_signal_actions'],0)
        self.assertTrue(all(result[key]==0 for key in entry.original.ZERO_FIELDS))
        self.assertTrue(all(result[key] is None for key in entry.original.FORMAL))
        self.assertIn('three saved training',result['scope']);self.assertNotIn('metadata-only',result['scope'])
        row = entry.original.read(self.args.registry)
        self.assertEqual(set(row),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        self.assertEqual(row['round'],entry.ROUND);self.assertEqual(row['model'],entry.original.MODEL)
        self.assertTrue(row['metrics']['model_compiled_accepted'])
        marker=entry.original.read(self.args.out/'completion.json')
        self.assertEqual(marker['status'],'ok');self.assertTrue(marker['registry_persisted'])
        self.assertTrue(marker['required_artifacts_persisted'])
        for name in ('export_math','native_compile'):
            self.assertEqual(entry.original.read(self.args.out/'phases'/name/'process-return.json'),clean())
        self.assertTrue((self.args.out/'source/heir-vgg-cpu-overlap.py').is_file())
        self.assertTrue((self.args.out/'source/shared-cpu-overlap-guard.py').is_file())

    def test_private_namespace_preserves_locks_algorithms_and_old_globals(self):
        before=entry.original.execute.__globals__['ROUND']
        namespace=entry._namespace(self.args)
        self.assertIs(namespace['execute'].__code__,entry.original.execute.__code__)
        self.assertIs(namespace['serial_lock'],entry.original.serial_lock)
        for name in ('worker','worker_command','native_command','check_worker','check_native','run_probes'):
            self.assertIs(namespace[name],getattr(entry.original,name))
        self.assertEqual(entry.original.execute.__globals__['ROUND'],before)
        with self.assertRaises(TypeError):entry.overlap._clone_plain(entry.original.serial_lock,namespace)

    def test_guard_replaces_only_two_rss_reasons_and_preserves_other_failures(self):
        raw=dict(stop=True,reasons=list(entry.RSS4_REASONS)+['arbitrary unknown guard reason'],scope='shared',subtree_rss_bytes=7*entry.GIB)
        with patch.object(entry.overlap,'parallel_observation',return_value=raw):
            row=self.observe()
        self.assertEqual(row['reasons'],['arbitrary unknown guard reason']);self.assertTrue(row['stop'])
        self.assertEqual(set(row['shared_guard_reasons_before_RSS_policy']),entry.RSS4_REASONS|{'arbitrary unknown guard reason'})
        self.assertEqual(row['shared_guard_scope'],'shared')

    def test_real_aggregate_rss10_available2_and_disk10_are_independent(self):
        self.assertFalse(self.observe(dict(subtree_rss_bytes=10*entry.GIB-2*1024**2),2*entry.GIB,10*entry.GIB)['stop'])
        self.assertTrue(self.observe(dict(subtree_rss_bytes=10*entry.GIB),2*entry.GIB,10*entry.GIB)['stop'])
        self.assertTrue(self.observe(available=2*entry.GIB-1)['stop'])
        self.assertTrue(self.observe(free=10*entry.GIB-1)['stop'])
        self.table[78]=dict(pid=78,ppid=os.getpid(),starttime=6,state='R',vmrss_kib=11*1024**2)
        self.assertTrue(self.observe()['stop'])

    def test_exact_bert_birth_lease_namespace_and_foreign_compiler_reject(self):
        self.assertFalse(self.observe()['stop'])
        self.table[self.args.bert_pid]['starttime']+=1;self.assertTrue(self.observe()['stop'])
        self.table=self.process_table()
        with patch.object(entry.overlap,'_lease_owners',return_value=[999]):self.assertTrue(self.observe()['stop'])
        self.table[77]=dict(pid=77,ppid=os.getpid(),starttime=8,state='R');self.names[77]='heir-opt'
        self.assertFalse(self.observe()['stop'])
        self.table[78]=dict(pid=78,ppid=1,starttime=9,state='R');self.names[78]='heir-opt'
        self.assertTrue(self.observe()['stop'])
        del self.table[78];del self.table[os.getpid()];self.assertTrue(self.observe()['stop'])

    def test_initial_bert_phase_change_closes_without_model_or_compile(self):
        put(self.args.bert_case/'session/SESSION.json',dict(phase='confirm',iteration=3,calls=3,failed=None))
        result=self.fake()
        self.assertEqual(result['status'],'resource_stopped');self.assertEqual(self.calls,[])
        self.assertEqual(result['attempted'],dict(export_math=0,native_compile=0))
        self.assertFalse(result['model_compiled_accepted']);self.assertTrue((self.args.out/'completion.json').is_file())

    def test_original_source_failure_persists_without_children(self):
        result=self.fake(source_failure=True)
        self.assertEqual(result['status'],'error');self.assertEqual(self.calls,[])
        self.assertIn('synthetic source rejected',result['error'])
        self.assertTrue((self.args.out/'completion.json').is_file())

    def test_terminal_bert_transition_preserves_compiled_fact_and_revokes_acceptance(self):
        def compiled(*unused):
            put(self.args.bert_case/'session/SESSION.json',dict(phase='done',iteration=3,calls=3,failed=None))
            return {'static_bootstrap_operations':2}
        result=self.fake(native_check=compiled)
        self.assertTrue(result['model_compiled']);self.assertEqual(result['completed']['native_compile'],1)
        self.assertEqual(result['status'],'resource_stopped');self.assertFalse(result['model_compiled_accepted'])
        self.assertFalse(entry.original.read(self.args.registry)['metrics']['model_compiled_accepted'])
        self.assertTrue(entry.original.read(self.args.registry)['metrics']['model_compiled'])

    def test_late_terminal_case_allowance_error_revokes_acceptance(self):
        real=entry.original.artifact_bytes
        compiled=[False]
        def size(path):
            return entry.original.CASE_BYTES if compiled[0] else real(path)
        def native(*unused):
            compiled[0]=True
            return {'static_bootstrap_operations':2}
        with patch.object(entry.original,'artifact_bytes',side_effect=size):
            result=self.fake(native_check=native)
        self.assertTrue(result['model_compiled']);self.assertEqual(result['status'],'resource_stopped')
        self.assertFalse(result['model_compiled_accepted'])
        self.assertFalse(entry.original.read(self.args.registry)['metrics']['model_compiled_accepted'])

    def test_nonclean_child_return_preserved_and_native_not_started(self):
        result=self.fake(bad={'remaining_pids':[777]})
        self.assertEqual(len(self.calls),1);self.assertFalse(result['model_compiled'])
        self.assertEqual(result['completed'],dict(export_math=0,native_compile=0))
        self.assertEqual(entry.original.read(self.args.out/'phases/export_math/process-return.json')['remaining_pids'],[777])

    def test_post_return_guard_retains_observation_and_stops_next_stage(self):
        def transition(_):
            put(self.args.bert_case/'session/SESSION.json',dict(phase='done',iteration=3,calls=3,failed=None))
        result=self.fake(after=transition)
        self.assertEqual(result['status'],'resource_stopped');self.assertEqual(len(self.calls),1)
        self.assertEqual(result['phases']['export_math']['observation'],clean())
        self.assertEqual(entry.original.read(self.args.out/'phases/export_math/process-return.json'),clean())

    def test_monitor_quarter_second_or_earlier_only_self_stop(self):
        stopped=threading.Event(); signals=[]
        def process(command,options):
            put(self.args.bert_case/'session/SESSION.json',dict(phase='done',iteration=3,calls=3,failed=None))
            self.assertTrue(stopped.wait(2))
            return clean(signal=signal.SIGTERM)
        def kill(pid,signum):
            signals.append((pid,signum));stopped.set()
        old=signal.getsignal(signal.SIGTERM)
        with patch.object(entry.original.os,'kill',side_effect=kill):result=self.fake(process=process)
        self.assertEqual(signals,[(os.getpid(),signal.SIGTERM)]);self.assertIs(signal.getsignal(signal.SIGTERM),old)
        self.assertFalse(result['model_compiled_accepted']);self.assertTrue((self.args.out/'completion.json').is_file())

    def test_original_guard_log_failure_still_self_stops(self):
        stopped=threading.Event();signals=[]
        durable=entry.original.durable_json
        def write(path,value):
            if Path(path).name=='monitor-stop.json':raise OSError('synthetic guard log failure')
            return durable(path,value)
        def process(command,options):
            put(self.args.bert_case/'session/SESSION.json',dict(phase='done',iteration=3,calls=3,failed=None))
            self.assertTrue(stopped.wait(2));return clean(signal=signal.SIGTERM)
        def kill(pid,signum):signals.append((pid,signum));stopped.set()
        errors=[]
        with patch.object(entry.original,'durable_json',side_effect=write),patch.object(entry.original.os,'kill',side_effect=kill),\
                patch.object(threading,'excepthook',side_effect=lambda args:errors.append(args.exc_value)):
            result=self.fake(process=process)
        self.assertEqual(signals,[(os.getpid(),signal.SIGTERM)]);self.assertEqual(len(errors),1)
        self.assertFalse(result['model_compiled_accepted']);self.assertTrue((self.args.out/'completion.json').is_file())

    def test_prepare_has_zero_model_and_compile_and_never_accepts(self):
        self.args.run=False;result=self.fake()
        self.assertEqual(result['status'],'prepared');self.assertEqual(self.calls,[])
        self.assertFalse(result['model_compiled']);self.assertFalse(result['model_compiled_accepted'])
        self.assertIn(entry.original.MODULE,result['planned_worker_command'])

    def test_single_attempt_registry_gate_retains_exact_worker_source_object(self):
        with patch.object(entry.original,'source_gate',return_value=self.checked):
            self.assertIs(entry.source_gate(self.source,self.args),self.checked)
            put(self.args.registry,dict(round=entry.ROUND,mode=entry.MODE,metrics=dict(attempted=dict(export_math=1))))
            with self.assertRaisesRegex(ValueError,'already attempted'):entry.source_gate(self.source,self.args)

    def test_cli_only24_and_positive_distinct_identities(self):
        cli=['--out',str(self.args.out),'--bert-case',str(self.args.bert_case),'--bert-run-case',str(self.args.bert_run_case),
             '--batch-case',str(self.args.batch_case),'--bert-pid','64170','--bert-starttime','1157046',
             '--batch-pid','36762','--batch-starttime','961036']
        self.assertEqual(entry.parser().parse_args(cli).memory_gib,24)
        with redirect_stderr(StringIO()),self.assertRaises(SystemExit):entry.parser().parse_args(cli+['--memory-gib','12'])
        for name,value in [('memory_gib',12),('bert_pid',True),('batch_starttime',0),('out',self.args.bert_case)]:
            bad=Namespace(**vars(self.args));setattr(bad,name,value)
            with self.subTest(name=name),self.assertRaises(ValueError):entry.execute(bad)
        self.assertFalse(self.args.out.exists())

    def test_original_execute_keeps_intake_venv_symlink_with_host_invisible_target(self):
        interpreter=self.root/'original-intake-venv/bin/python'
        interpreter.parent.mkdir(parents=True)
        missing_target=self.root/'host-invisible-nix-store/python'
        interpreter.symlink_to(missing_target)
        self.args.python=interpreter
        tool=self.args.heir_opt
        alias=self.root/'compiler-alias';alias.symlink_to(tool)
        self.args.heir_opt=alias
        self.assertTrue(interpreter.is_symlink());self.assertFalse(interpreter.is_file())
        result=self.fake()
        self.assertEqual(result['status'],'ok')
        self.assertEqual(self.args.python,interpreter.absolute())
        self.assertNotEqual(self.args.python,missing_target.resolve())
        worker=self.calls[0][0]
        self.assertEqual(worker[worker.index('--python')+1],str(interpreter.absolute()))
        self.assertEqual(result['planned_worker_command'],worker)
        self.assertEqual(self.calls[1][0][0],str(tool.resolve()))


if __name__=='__main__':
    unittest.main()
