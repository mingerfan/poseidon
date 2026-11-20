"""Small actual-schema queue, source, supervision and persistence fixtures."""
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_ordered_mock_pair as entry


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def closed(root, name, result):
    case = root/name
    write(case/'result.json', result)
    write(case/'completion.json', dict(status=result['status'], round=result['round'],
        required_artifacts_persisted=True, registry_persisted=True, started_ns=100,
        ended_ns=5399*10**9+100, complete_case_elapsed_seconds=5399))
    return case


def fixture(root):
    args = SimpleNamespace(source_case=root/'old-c2', compile_case=root/'old-master',
        first_case=root/'first-fresh', prior_case=root/'actual-owner', out=root/'new-pair',
        metadata_template=root/'existing-template.json', intake_python=Path('/existing/intake'),
        data_python=Path('/existing/data'), ledger=root/'ledger.jsonl', allowance=root/'allowance.json',
        host_pid_namespace=os.readlink('/proc/self/ns/pid'), prior_pid=9999999, prior_starttime=7,
        deadline_timestamp=time.time()+86400)
    first = dict(round=entry.fresh.ROUND, mode=entry.fresh.MODE, model=entry.MODEL,
        status='error', compile_accepted=False, metrics={'status': 'error'},
        wall_policy=dict(search_absolute_ceiling_seconds=5400, total_seconds=7200),
        attempted=dict(confirm=0, math=0, emit=0, trace=0, compile=0),
        processes=[dict(phase='step', argv=['python', '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '5221'],
            observation=dict(exit_code=124, timed_out=False, signal=None, signals_sent=[],
                             remaining_pids=[], started_ns=100, ended_ns=5221*10**9+100))])
    closed(root, args.first_case.name, first)
    write(args.first_case/'two-hour-record.json', dict(round=entry.fresh.ROUND, mode=entry.fresh.MODE,
        model=entry.MODEL, compile_accepted=False, persisted_pipeline_within_two_hours=False,
        started_ns=100, required_artifacts_and_registry_completed_ns=5399*10**9+100))
    process = dict(exit_code=1, timed_out=False, signal=None, remaining_pids=[], signals_sent=[])
    jobs = []
    for name, module, case, seconds in (
        ('fresh', entry.fresh.MODULE, args.first_case, 7320),
        ('native', 'pt_agent.experiments.bert_rule63_shared_optimized_compile', root/'native-current', 9720)):
        argv = ['python', '-B', '-m', module, '--out', str(case)]
        jobs.append(dict(id=name, out=str(case), argv=argv, seconds=seconds,
                         process=copy.deepcopy(process), status='failed', cleanup_unknown=False))
        write(args.prior_case/(name+'-command.json'), argv)
    # This is the real serial result shape: no top-level model or metrics.
    owner = dict(round='AUTONOMOUS-SERIAL', mode='finite-trusted-serial-experiments',
                 status='completed_with_observations', jobs=jobs)
    closed(root, args.prior_case.name, owner)
    write(args.prior_case/'manifest.json', dict(jobs=jobs))
    write(args.prior_case/'supervisor.json', dict(pid=args.prior_pid,
                                               process=dict(pid=args.prior_pid, starttime=args.prior_starttime)))
    native=dict(round='CURRENT-NATIVE',mode='current-native-only',model=entry.MODEL,
                status='error',metrics={'status':'error'})
    closed(root,'native-current',native)
    args.compile_case.mkdir()
    write(args.compile_case/'outer-admission.json', {'existing': True})
    reference = dict(native_case=str(root/'old-native'), program=str(args.source_case/'session/program.ptdsl'),
        model=str(root/'official-model'), inputs=str(root/'original-sst2'),
        source=dict(checkpoint_audit=str(root/'original-checkpoint/result.json'), fork_provenance=dict(
            original_source_case=str(root/'original-source'), continuation_case=str(root/'old-continuation'))))
    rows = [dict(run_id=args.prior_case.name, round=owner['round'], mode=owner['mode'],
                 model='original-model-experiment-queue', git_rev=None,
                 metrics=dict(status=owner['status'], jobs=jobs), artifacts_dir=str(args.prior_case), ts='fixture'),
            dict(run_id=args.first_case.name, round=first['round'], mode=first['mode'], model=entry.MODEL,
                 git_rev=None, metrics=first['metrics'], artifacts_dir=str(args.first_case), ts='fixture'),
            dict(run_id='native-current',round=native['round'],mode=native['mode'],model=entry.MODEL,
                 git_rev=None,metrics=native['metrics'],artifacts_dir=str(root/'native-current'),ts='fixture')]
    (root/'registry.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return args, reference


def good_worker(args, proof):
    return dict(schema='bert-ordered-original-mock-pair-result-v1', status='ok', pair_accepted=True,
        ordered_execution_observed=True, source=proof, source_admission=str(args.out/'source-admission.json'),
        attempted=dict(tensor=1, original_mock=1, ordered_mock=1),
        completed=dict(tensor=1, original_mock=1, ordered_mock=1), logical_heldout_rows=40,
        original_mock_batch_size=10, requested_workers=3, cold_chain_measurement=False,
        qualifies_two_hour_claim=False, speedup=None,
        pair_checks=dict(output_mathematics=dict(passed=True, max_absolute_error=0., max_tolerance_ratio=0.,
            criterion=entry.reference.compiled.base.original.CRITERION), full_metrics_equal=True,
            original_domain_diagnostics_equal=True, original_event_indices_equal=True),
        **dict.fromkeys(entry.ZERO, 0))


class OuterPairTests(unittest.TestCase):
    def test_existing_venv_interpreter_path_keeps_its_environment_identity(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);args,_=fixture(root)
            base=root/'base-python';base.write_text('presence-only interpreter fixture')
            venv=root/'venv/bin/python';venv.parent.mkdir(parents=True);venv.symlink_to(base)
            args.intake_python=args.data_python=venv
            entry.normalize_args(args)
            self.assertEqual(args.intake_python,venv.absolute())
            self.assertEqual(args.data_python,venv.absolute())
            self.assertNotEqual(args.intake_python,base.resolve())

    def test_optional_data_reader_is_inferred_from_original_source_record(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);args,_=fixture(root)
            reader=root/'existing-data-venv/bin/python'
            write(args.source_case/'model/samples/source.json',dict(reader_python=str(reader)))
            args.data_python=None
            entry.normalize_args(args)
            self.assertEqual(args.data_python,reader.absolute())

    def test_actual_model_less_queue_schema_and_failed_jobs_can_close(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);args,_=fixture(root)
            with patch.object(entry.paths, 'RESULTS_DIR', root), patch.object(entry, '_process', return_value=None):
                proof=entry.owner_gate(args)
            self.assertEqual(proof['actual_jobs_closed'],2)
            self.assertEqual(proof['status'],'completed_with_observations')

    def test_live_original_birth_including_zombie_is_rejected(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);args,_=fixture(root)
            with patch.object(entry.paths,'RESULTS_DIR',root), patch.object(entry,'_process',return_value={
                    'pid':args.prior_pid,'starttime':args.prior_starttime,'state':'Z'}), self.assertRaises(ValueError):
                entry.owner_gate(args)

    def test_owner_registry_disagreement_and_shortened_native_budget_rejected(self):
        for fault in ('registry','budget'):
            with self.subTest(fault=fault), TemporaryDirectory() as folder:
                root=Path(folder);args,_=fixture(root)
                rows=[json.loads(line) for line in(root/'registry.jsonl').read_text().splitlines()]
                if fault=='registry':rows[0]['metrics']['status']='ok'
                else:
                    owner=entry.read(args.prior_case/'result.json')
                    owner['jobs'][1]['argv'] += ['--native-seconds','3600']
                    write(args.prior_case/'result.json',owner)
                    write(args.prior_case/'manifest.json',dict(jobs=owner['jobs']))
                    write(args.prior_case/'native-command.json',owner['jobs'][1]['argv'])
                    rows[0]['metrics']['jobs']=owner['jobs']
                (root/'registry.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                with patch.object(entry.paths,'RESULTS_DIR',root), patch.object(entry,'_process',return_value=None),self.assertRaises(ValueError):
                    entry.owner_gate(args)

    def test_complete_old_source_admission_and_all_birth_families_required(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);args,original=fixture(root)
            with patch.object(entry.paths,'RESULTS_DIR',root), patch.object(entry,'_process',return_value=None),\
                 patch.object(entry.reference,'admission',return_value=original),\
                 patch.object(entry,'producer_births',return_value=2) as births:
                proof=entry.source_gate(args)
            expected={args.source_case,args.compile_case,args.prior_case,args.first_case,root/'native-current',
                      root/'old-native',root/'original-source',root/'old-continuation'}
            self.assertEqual({call.args[0] for call in births.call_args_list},expected)
            self.assertTrue(proof['all_source_births_closed'])
            self.assertFalse(proof['report_read_capability'])
            self.assertEqual(proof['logical_heldout_rows'],40)
            self.assertFalse(args.out.exists())

    def test_open_source_producers_rejected_and_old_outer_proof_must_exist(self):
        for fault in ('birth','old_proof'):
            with self.subTest(fault=fault),TemporaryDirectory() as folder:
                root=Path(folder);args,original=fixture(root)
                if fault=='old_proof':(args.compile_case/'outer-admission.json').unlink()
                with patch.object(entry.paths,'RESULTS_DIR',root),patch.object(entry,'_process',return_value=None),\
                     patch.object(entry.reference,'admission',return_value=original),\
                     patch.object(entry,'producer_births',side_effect=ValueError('still live')),self.assertRaises(ValueError):
                    entry.source_gate(args)

    def test_new_directory_and_full_absolute_deadline_precede_creation(self):
        for fault in ('nested','deadline'):
            with self.subTest(fault=fault),TemporaryDirectory() as folder:
                root=Path(folder);args,_=fixture(root)
                if fault=='nested':args.out=args.source_case/'new'
                else:args.deadline_timestamp=time.time()+1
                with patch.object(entry.paths,'RESULTS_DIR',root),self.assertRaises(ValueError):entry.execute(args)
                self.assertFalse(args.out.exists())

    def test_explicit_original_pure_worker_arguments_and_zero_report_scope(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);args,_=fixture(root)
            proof={key:str(root/key) for key in('model_dir','program','frozen','model','inputs','metadata_template')}
            argv=entry.command(args,proof,1800)
            self.assertEqual(argv[2:4],['-m','pt_agent.tools.nixrun'])
            self.assertEqual(entry.option(argv,'--seconds'),'1800')
            self.assertIn(entry.WORKER,argv)
            self.assertNotIn('--ledger',argv)
            self.assertNotIn('--phase',argv)
            self.assertNotIn('report',argv)

    def test_worker_scope_complete_math_and_no_warm_two_hour_substitution(self):
        with TemporaryDirectory() as folder:
            args,_=fixture(Path(folder));proof={'source':'fixture'};value=good_worker(args,proof)
            entry.check_worker(args,value,proof)
            for path,update in (('qualifies_two_hour_claim',True),('compiler_runs',1),('requested_workers',1)):
                changed=copy.deepcopy(value);changed[path]=update
                with self.subTest(path=path),self.assertRaises(ValueError):entry.check_worker(args,changed,proof)
            value['pair_checks']['output_mathematics']['max_tolerance_ratio']=1.01
            with self.assertRaises(ValueError):entry.check_worker(args,value,proof)

    def test_sampled_peak_and_actual_child_birth_closure(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);args,_=fixture(root);args.out.mkdir()
            sample=dict(outer_process={'pid':os.getpid(),'vmrss_kib':100},
                        processes=[dict(pid=9876543,starttime=9,vmrss_kib=200)])
            path=args.out/'worker-resources.jsonl';path.write_text(json.dumps(sample)+'\n')
            with patch.object(entry,'_process',return_value=None):result=entry.sampled_resources(args)
            self.assertEqual(result['max_sampled_subtree_rss_gib'],300/1024**2)
            self.assertTrue(result['sampled_rss_policy_passed'])
            with patch.object(entry,'_process',return_value={'starttime':9}),self.assertRaises(ValueError):entry.sampled_resources(args)
            with patch.object(entry,'_process',side_effect=PermissionError('unproven original handle')),self.assertRaises(PermissionError):entry.sampled_resources(args)
            path.write_text(json.dumps(sample))
            with self.assertRaises(ValueError):entry.sampled_resources(args)

    def test_successful_supervision_persists_registry_and_masks_terminal_signals(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);args,_=fixture(root)
            proof={key:str(root/key) for key in('model_dir','program','frozen','model','inputs','metadata_template')}
            @contextmanager
            def lease(*unused):yield
            def runner(argv,**unused):
                write(args.out/'worker/pair-result.json',good_worker(args,proof))
                return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])
            original_register=entry.register;seen=[]
            def register(*values):
                signal.raise_signal(signal.SIGTERM)
                signal.raise_signal(signal.SIGINT)
                seen.append(True)
                return original_register(*values)
            handlers={sig:signal.getsignal(sig) for sig in(signal.SIGTERM,signal.SIGINT)}
            with patch.object(entry.paths,'RESULTS_DIR',root),patch.object(entry,'source_gate',return_value=proof),\
                 patch.object(entry,'heavy_lease',side_effect=lease),patch.object(entry.reference.serial,'foreign_heavy_processes',return_value=[]),\
                 patch.object(entry,'resources',return_value={'stop':False,'reasons':[]}),\
                 patch.object(entry,'sampled_resources',return_value={'complete_samples':True,'sampled_rss_policy_passed':True}),\
                 patch.object(entry,'run_process',side_effect=runner),patch.object(entry,'register',side_effect=register):
                result=entry.execute(args)
            self.assertEqual(result['status'],'ok')
            self.assertTrue(result['pair_accepted'])
            self.assertTrue(result['monitor_stopped'])
            self.assertEqual(seen,[True])
            self.assertEqual({sig:signal.getsignal(sig) for sig in handlers},handlers)
            completion=entry.read(args.out/'completion.json');self.assertTrue(completion['registry_persisted'])
            rows=[json.loads(x) for x in(root/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len([r for r in rows if r['run_id']==args.out.name]),1)

    def test_complete_numeric_outputs_do_not_override_excess_sampled_rss(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);args,_=fixture(root)
            proof={key:str(root/key) for key in('model_dir','program','frozen','model','inputs','metadata_template')}
            @contextmanager
            def lease(*unused):yield
            def runner(argv,**unused):
                write(args.out/'worker/pair-result.json',good_worker(args,proof))
                return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])
            observation=dict(complete_samples=True,max_sampled_subtree_rss_gib=14.01,sampled_rss_policy_passed=False)
            with patch.object(entry.paths,'RESULTS_DIR',root),patch.object(entry,'source_gate',return_value=proof),\
                 patch.object(entry,'heavy_lease',side_effect=lease),patch.object(entry.reference.serial,'foreign_heavy_processes',return_value=[]),\
                 patch.object(entry,'resources',return_value={'stop':False,'reasons':[]}),\
                 patch.object(entry,'sampled_resources',return_value=observation),patch.object(entry,'run_process',side_effect=runner):
                result=entry.execute(args)
            self.assertEqual(result['status'],'error');self.assertFalse(result['pair_accepted'])
            self.assertTrue(result['worker']['pair_accepted'])
            self.assertEqual(result['resource_sampling'],observation)

    def test_resource_guard_stops_owned_run_even_when_guard_record_write_fails(self):
        for write_fails in (False,True):
            with self.subTest(write_fails=write_fails),TemporaryDirectory() as folder:
                root=Path(folder);args,_=fixture(root)
                proof={key:str(root/key) for key in('model_dir','program','frozen','model','inputs','metadata_template')}
                @contextmanager
                def lease(*unused):yield
                calls=[]
                def resources(*unused):
                    calls.append(True)
                    return dict(stop=len(calls)>1,reasons=['tiny fixture guard'] if len(calls)>1 else [])
                def runner(*unused,**ignored):
                    time.sleep(2)
                    self.fail('Owned guard did not interrupt its fake run')
                original_save=entry.save
                def save(path,value):
                    if write_fails and path.name=='guard-stop.json':raise OSError('tiny fixture disk failure')
                    return original_save(path,value)
                with patch.object(entry.paths,'RESULTS_DIR',root),patch.object(entry,'source_gate',return_value=proof),\
                     patch.object(entry,'heavy_lease',side_effect=lease),\
                     patch.object(entry.reference.serial,'foreign_heavy_processes',return_value=[]),\
                     patch.object(entry,'resources',side_effect=resources),patch.object(entry,'run_process',side_effect=runner),\
                     patch.object(entry,'save',side_effect=save):
                    result=entry.execute(args)
                self.assertEqual(result['status'],'error')
                self.assertFalse(result['pair_accepted'])
                self.assertTrue(result['monitor_stopped'])
                self.assertEqual(result['attempted_pair'],1)
                self.assertEqual(bool(result['guard_persistence_error']),write_fails)
                self.assertEqual(entry.read(args.out/'completion.json')['status'],'error')


if __name__=='__main__':unittest.main()
