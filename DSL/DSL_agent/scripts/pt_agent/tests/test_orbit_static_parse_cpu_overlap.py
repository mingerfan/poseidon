"""New metadata fixtures/fake subprocesses; no Go, model, sample or API calls."""
from argparse import Namespace
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import orbit_static_parse_cpu_overlap as entry


class StaticOverlapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = self.fixture()
        self.names = {}

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False)+'\n')

    @staticmethod
    def clean(**changes):
        result = dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])
        result.update(changes)
        return result

    def fixture(self, model='resnet20'):
        args = Namespace(model=model, out=self.root/('orbit-'+model+'-static-parse-cpu-overlap-fixture'),
            source_case=self.root/'optimizer', shared_build_case=self.root/'shared',
            orbit_root=self.root/'orbit', go_sdk=self.root/'sdk', initialization_audit=self.root/'init.json',
            bert_case=self.root/'bert-source', bert_run_case=self.root/'bert-run', batch_case=self.root/'batch',
            bert_pid=64170, bert_starttime=1157046, batch_pid=36762, batch_starttime=961036,
            registry=self.root/'registry.jsonl', run=True)
        self.write(args.bert_case/'session/SESSION.json', dict(phase='tune', iteration=3, calls=3, failed=None))
        self.write(args.bert_run_case/'result.json', dict(round='BERT-SERIAL-CONTINUATION', model='bert-tiny-sst2',
            status='running', termination=None, source=dict(source_case=str(args.bert_case)),
            attempted=dict(step=1), completed=dict(step=0)))
        self.write(args.batch_case/'supervisor.json', dict(pid=args.batch_pid,
            process=dict(pid=args.batch_pid, starttime=args.batch_starttime)))
        self.write(args.batch_case/'active.json', dict(id='original-bert', status='running', out=str(args.bert_run_case),
            argv=['python3', '-B', '-m', 'pt_agent.experiments.bert_serial_continuation',
                  '--source-case', str(args.bert_case)]))
        (self.root/'autonomous-heavy-experiments.lock').touch()
        module=args.orbit_root/'backend/lattigo'
        for name in ('parser.go', 'scheme.go', 'exeorder.go'):
            path=module/'lowering/fhe'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('package main\n')
        (args.shared_build_case/'work/module-cache').mkdir(parents=True, exist_ok=True)
        (args.shared_build_case/'work/module-cache/public.go').write_text('package public\n')
        self.write(args.initialization_audit, dict(public_fixture=True))
        return args

    def table(self):
        return {self.args.batch_pid:dict(pid=self.args.batch_pid, starttime=self.args.batch_starttime, state='S', ppid=1),
            40000:dict(pid=40000, starttime=5, state='S', ppid=self.args.batch_pid),
            self.args.bert_pid:dict(pid=self.args.bert_pid, starttime=self.args.bert_starttime, state='R', ppid=40000),
            os.getpid():dict(pid=os.getpid(), starttime=9, state='R', ppid=1)}

    def patches(self):
        stack=ExitStack()
        stack.enter_context(patch.object(entry.paths, 'RESULTS_DIR', self.root))
        stack.enter_context(patch.object(entry, '_scan', side_effect=self.table))
        stack.enter_context(patch.object(entry, '_process_name', side_effect=lambda pid:self.names.get(pid, 'python3')))
        stack.enter_context(patch.object(entry, '_lease_owners', return_value=[self.args.batch_pid]))
        self.addCleanup(stack.close)
        return stack

    def observe(self, **changes):
        options=dict(sample=None, available=12*entry.GIB, free=20*entry.GIB)
        options.update(changes)
        return entry.parallel_observation(self.args, **options)

    def checked(self):
        factory=entry.original.shared.MODEL if self.args.model=='resnet20' else entry.common.MODEL_RECORDS['vgg16']['factory']
        return dict(optimized_source=dict(model=factory, optimized_mlir=dict(path=str(self.root/'saved.mlir'), bytes=19),
                optimizer_params=dict(max_slot=65536, bts_ub=16)),
            expected_graph=dict(node_count=3, edge_count=2, inputs=['arg0'], outputs=['1'], output_id=1,
                op_counts=dict(input=1, constant=1, add=1), placement_bootstrap_nodes=0),
            module=str(self.args.orbit_root/'backend/lattigo'),
            module_cache=str(self.args.shared_build_case/'work/module-cache'))

    def parsed(self):
        checked=self.checked()
        return dict(status='ok', mlir_path=checked['optimized_source']['optimized_mlir']['path'], mlir_bytes=19,
            mlir_text_reads=2, parsed_nodes=3, parsed_operations=2, input_count=1, output_id=1, edge_count=2,
            op_counts=checked['expected_graph']['op_counts'], static_bootstrap_nodes=0, topological_order_count=2,
            nodes=[dict(id=0), dict(id=1)], official_ReadFile_calls=1, official_parseMLIROperation_calls=2,
            official_getOptimalRunOrder_calls=1, official_getMaxWorkingPoolSize_calls=1,
            **dict.fromkeys(entry.ZERO_FIELDS[:9], 0))

    def outer(self, *, failure=None, callback=None):
        self.patches()
        calls=[]
        def process(command, **options):
            phase=Path(options['log']).stem
            calls.append((phase, command, options))
            Path(options['log']).write_text('fake original parser process\n')
            Path(options['resources']).write_text('')
            if phase=='build':
                (self.args.out/'orbit-parser').write_bytes(b'public fake binary')
            elif phase=='selftest':
                self.write(self.args.out/'selftest.json', dict(status='ok', selftests=6, model_parses=0,
                    **dict.fromkeys(entry.ZERO_FIELDS[:9], 0)))
            else:
                row=self.parsed()
                if failure=='graph':row['edge_count']=8
                self.write(self.args.out/'parsed-model.json', row)
            if callback:callback(phase)
            return self.clean(exit_code=1 if failure==phase else 0)
        with patch.object(entry.original, 'source_gate', side_effect=lambda args:self.checked()), \
                patch.object(entry.common, 'source_gate', side_effect=lambda args:self.checked()), \
                patch.object(entry.original, '_available_ram_bytes', return_value=12*entry.GIB), \
                patch.object(entry.original.shutil, 'disk_usage', return_value=SimpleNamespace(free=30*entry.GIB)), \
                patch.object(entry.original, 'run_process', side_effect=process):
            result=entry.execute(self.args)
        return result,calls

    def test_original_session_and_actual_lease_owner_are_required(self):
        self.patches()
        row=self.observe()
        self.assertFalse(row['stop'])
        self.assertEqual(row['bert_and_batch']['calls'],3)
        self.assertEqual(row['heavy_lease_owners'],[36762])
        with patch.object(entry, '_lease_owners', return_value=[999]):
            self.assertTrue(self.observe()['stop'])

    def test_session_phase_iteration_calls_and_failure_stop(self):
        self.patches()
        path=self.args.bert_case/'session/SESSION.json'
        original=entry.read(path)
        for field,value in (('phase','confirm'),('iteration',4),('iteration',True),('calls',4),('failed','error')):
            self.write(path,dict(original,**{field:value}))
            with self.subTest(field=field,value=value):self.assertTrue(self.observe()['stop'])
        self.write(path,original)

    def test_identity_reuse_missing_namespace_and_wrong_ancestry_stop(self):
        self.patches()
        for change in ('missing','birth','state','ancestry'):
            table=self.table()
            if change=='missing':del table[self.args.bert_pid]
            elif change=='birth':table[self.args.bert_pid]['starttime']+=1
            elif change=='state':table[self.args.bert_pid]['state']='Z'
            else:table[self.args.bert_pid]['ppid']=1
            with self.subTest(change=change),patch.object(entry,'_scan',return_value=table):
                self.assertTrue(self.observe()['stop'])

    def test_original_batch_active_source_and_terminal_state_are_bound(self):
        self.patches()
        path=self.args.batch_case/'active.json';original=entry.read(path)
        self.write(path,dict(original,out=str(self.root/'foreign')))
        self.assertTrue(self.observe()['stop'])
        self.write(path,original)
        self.write(self.args.batch_case/'result.json',dict(status='ok'))
        self.assertTrue(self.observe()['stop'])

    def test_foreign_compilers_stop_but_own_go_descendants_are_excluded(self):
        self.patches();table=self.table()
        table[8]=dict(pid=8,ppid=os.getpid(),starttime=3,state='R');self.names[8]='compile'
        with patch.object(entry,'_scan',return_value=table):self.assertFalse(self.observe()['stop'])
        table[9]=dict(pid=9,ppid=1,starttime=4,state='R');self.names[9]='hecate-opt'
        with patch.object(entry,'_scan',return_value=table):
            self.assertEqual(self.observe()['foreign_compilers'],[dict(pid=9,starttime=4,name='hecate-opt')])

    def test_rss_and_host_available_guards_are_independent(self):
        self.patches()
        self.assertFalse(self.observe(sample=dict(subtree_rss_bytes=4*entry.GIB),available=2*entry.GIB)['stop'])
        self.assertTrue(self.observe(sample=dict(subtree_rss_bytes=4*entry.GIB+1))['stop'])
        self.assertTrue(self.observe(available=2*entry.GIB-1)['stop'])
        self.assertTrue(self.observe(free=10*entry.GIB-1)['stop'])

    def test_memory_guard_includes_supervisor_and_current_owned_descendants(self):
        self.patches();table=self.table()
        table[os.getpid()]['vmrss_kib']=40*1024
        table[8]=dict(pid=8,ppid=os.getpid(),starttime=3,state='R',vmrss_kib=4*1024**2)
        with patch.object(entry,'_scan',return_value=table):
            row=self.observe()
        self.assertTrue(row['stop'])
        self.assertEqual(row['supervisor_rss_bytes'],40*1024**2)
        self.assertEqual(row['subtree_rss_bytes'],4*entry.GIB+40*1024**2)

    def test_public_flock_identity_parser_matches_exact_device_inode(self):
        path=self.root/'autonomous-heavy-experiments.lock';stat=path.stat()
        device=f'{os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x}:{stat.st_ino}'
        text=f'1: FLOCK ADVISORY WRITE 36762 {device} 0 EOF\n2: -> FLOCK ADVISORY WRITE 9 {device} 0 EOF\n3: POSIX ADVISORY WRITE 9 {device} 0 EOF\n'
        self.assertEqual(entry._lease_owners(path,text),[36762])

    def test_private_plain_clones_preserve_original_contextmanager_and_client(self):
        namespace=entry._namespace(self.args)
        self.assertIs(namespace['CLIENT'],entry.original.CLIENT)
        self.assertIs(namespace['serial_lock'],entry.original.serial_lock)
        self.assertIs(namespace['check_parsed'],entry.original.check_parsed)
        self.assertIsNot(namespace['execute'].__globals__,entry.original.execute.__globals__)
        self.assertIs(entry.original.execute.__globals__['source_gate'],entry.original.source_gate)
        with self.assertRaises(TypeError):entry._clone_plain(entry.original.serial_lock,{})

    def test_original_model_source_route_and_parameter_contract(self):
        self.patches()
        for model in ('resnet20','vgg16'):
            self.args.model=model
            with patch.object(entry.original,'source_gate',return_value=self.checked()) as res, \
                    patch.object(entry.common,'source_gate',return_value=self.checked()) as vgg:
                row=entry.source_gate(self.args)
                self.assertEqual(res.call_count,int(model=='resnet20'))
                self.assertEqual(vgg.call_count,int(model=='vgg16'))
                self.assertFalse(row['global_heavy_lease_acquired_by_this_entry'])
        row=self.checked();row['optimized_source']['optimizer_params']['max_slot']=16384
        with patch.object(entry.common,'source_gate',return_value=row),self.assertRaises(ValueError):entry.source_gate(self.args)

    def test_full_fake_path_preserves_3phases_budgets_registry_and_completion(self):
        result,calls=self.outer()
        self.assertEqual(result['status'],'ok')
        self.assertTrue(result['backend_parser_consumption_verified'])
        self.assertEqual([x[0] for x in calls],['build','selftest','parse'])
        for (phase,command,options),seconds in zip(calls,(600,120,300)):
            self.assertEqual(options['mem_gib'],4)
            self.assertEqual(options['fsize_gib'],8)
            self.assertLessEqual(options['timeout'],seconds)
            self.assertTrue((self.args.out/(phase+'-process-return.json')).is_file())
        self.assertEqual(result['limits']['AS_gib'],4)
        self.assertEqual(result['limits']['subtree_RSS_gib'],4)
        self.assertEqual(result['limits']['minimum_available_gib'],2)
        self.assertEqual((self.args.out/'source/parser-main.go').read_text(),entry.CLIENT)
        registry=json.loads(self.args.registry.read_text())
        self.assertEqual(set(registry),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        self.assertEqual(registry['model'],entry.original.shared.MODEL)
        self.assertEqual(registry['round'],entry.ROUND)
        marker=entry.read(self.args.out/'completion.json')
        self.assertTrue(marker['registry_persisted']);self.assertTrue(marker['required_artifacts_persisted'])
        self.assertTrue(all(result[key]==0 for key in entry.ZERO_FIELDS))
        self.assertEqual(result['bert_signal_actions'],0)

    def test_vgg_target_registry_does_not_inherit_shared_res_model(self):
        self.args=self.fixture('vgg16');result,calls=self.outer()
        self.assertEqual(result['model'],entry.common.MODEL_RECORDS['vgg16']['factory'])
        self.assertEqual(json.loads(self.args.registry.read_text())['model'],result['model'])

    def test_first_failure_and_parsed_contract_rejection_preserve_evidence(self):
        result,calls=self.outer(failure='build')
        self.assertEqual(result['status'],'error');self.assertEqual(len(calls),1)
        self.assertEqual(result['completed']['build'],0)
        self.assertIsNone(result['backend_parser_consumption_verified'])
        self.assertTrue((self.args.out/'completion.json').is_file())

    def test_state_transition_after_process_stops_next_phase(self):
        def transition(phase):
            self.write(self.args.bert_case/'session/SESSION.json',dict(phase='confirm',iteration=3,calls=3,failed=None))
        result,calls=self.outer(callback=transition)
        self.assertNotEqual(result['status'],'ok');self.assertEqual(len(calls),1)
        self.assertEqual(result['attempted'],dict(build=1,selftest=0,parse=0))
        self.assertTrue((self.args.out/'build-process-return.json').is_file())
        self.assertIsNone(result['backend_parser_consumption_verified'])

    def test_continuous_monitor_signals_only_own_supervisor(self):
        signals=[];old=signal.getsignal(signal.SIGTERM)
        def changed(phase):
            if phase=='build':
                self.write(self.args.bert_case/'session/SESSION.json',dict(phase='done',iteration=3,calls=3,failed=None))
                until=time.monotonic()+2
                while not signals and time.monotonic()<until:time.sleep(.01)
        with patch.object(entry.original.os,'kill',side_effect=lambda pid,sig:signals.append((pid,sig))):
            result,calls=self.outer(callback=changed)
        self.assertTrue(signals);self.assertEqual(signals,[(os.getpid(),signal.SIGTERM)])
        self.assertNotEqual(result['status'],'ok');self.assertEqual(len(calls),1)
        self.assertEqual(result['bert_signal_actions'],0)
        self.assertIs(signal.getsignal(signal.SIGTERM),old)
        self.assertTrue((self.args.out/'monitor-stop.json').is_file())

    def test_preparation_runs_no_subprocess_and_prior_attempt_is_rejected(self):
        self.args.run=False;result,calls=self.outer()
        self.assertEqual(calls,[]);self.assertEqual(result['status'],'prepared')
        self.assertIsNone(result['backend_parser_consumption_verified'])
        row=json.loads(self.args.registry.read_text());row['metrics']['attempted']['build']=1
        self.write(self.args.registry,row)
        with patch.object(entry.original,'source_gate',return_value=self.checked()),self.assertRaisesRegex(ValueError,'once'):
            entry.source_gate(self.args)

    def test_fresh_independent_out_and_explicit_positive_identity(self):
        self.patches();self.args.out=self.root/'original-queued-case'
        with self.assertRaises(ValueError):entry.execute(self.args)
        for value in ('0','-1'):
            with self.subTest(value=value),self.assertRaises(Exception):entry.positive(value)
        self.assertEqual(entry.positive('64170'),64170)


if __name__=='__main__':
    unittest.main()
