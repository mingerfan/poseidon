"""Own tiny metadata/fake-supervision fixtures; original math/tools are not run."""
from argparse import Namespace
from contextlib import ExitStack
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_vgg_host_cpu_overlap as entry


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n'if isinstance(value,dict)else value)


def clean(**changes):
    result=dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[]);result.update(changes);return result


class HostOverlapChecks(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(entry.paths,'RESULTS_DIR',self.root))
        self.stack.enter_context(patch.object(entry.paths,'ORION_ROOT',self.root/'baselines/orion'))
        python=entry.host_python();put(python,'fake existing interpreter metadata only\n');python.chmod(0o700)
        self.args=Namespace(out=self.root/(entry.OUT_PREFIX+'fixture'),registry=self.root/'registry.jsonl',run=True,memory_gib=24,
            python=python,heir_opt=self.root/'heir-opt',bert_case=self.root/'bert-source',bert_run_case=self.root/'bert-run',
            batch_case=self.root/'batch',bert_pid=64170,bert_starttime=1157046,batch_pid=36762,batch_starttime=961036,
            **{name:None for name in entry.original.SOURCE_FIELDS})
        put(self.args.heir_opt,'fake native executable metadata only\n');self.args.heir_opt.chmod(0o700)
        put(self.args.bert_case/'session/SESSION.json',dict(phase='tune',iteration=3,calls=3,failed=None))
        put(self.args.bert_run_case/'result.json',dict(round='BERT-SERIAL-CONTINUATION',model='bert-tiny-sst2',status='running',
            termination=None,source=dict(source_case=str(self.args.bert_case)),attempted=dict(step=1),completed=dict(step=0)))
        put(self.args.batch_case/'supervisor.json',dict(pid=self.args.batch_pid,process=dict(pid=self.args.batch_pid,starttime=self.args.batch_starttime)))
        put(self.args.batch_case/'active.json',dict(status='running',id='original-bert',out=str(self.args.bert_run_case),
            argv=['python3','-B','-m','pt_agent.experiments.bert_serial_continuation','--source-case',str(self.args.bert_case)]))
        self.source={name:self.root/('original-'+name)for name in entry.original.SOURCE_FIELDS}
        put(self.source['native_parameters'],dict(logN=16,Q=[101],P=[103],logDefaultScale=45))
        self.checked=dict(contract=dict(role='fake-original'),native_parameters=dict(logN=16,Q=[101],P=[103],logDefaultScale=45),
            paths={name:str(path)for name,path in self.source.items()},
            pipeline='builtin.module(test{parameter-file='+str(self.source['native_parameters'])+'})')
        self.table={36762:dict(pid=36762,ppid=1,starttime=961036,state='S'),
            40000:dict(pid=40000,ppid=36762,starttime=5,state='S'),
            64170:dict(pid=64170,ppid=40000,starttime=1157046,state='R'),
            os.getpid():dict(pid=os.getpid(),ppid=1,starttime=9,state='R',vmrss_kib=2048)}
        self.available=12*entry.GIB;self.calls=[]
        self.stack.enter_context(patch.object(entry.overlap,'_scan',side_effect=lambda:self.table))
        self.stack.enter_context(patch.object(entry.overlap,'_lease_owners',return_value=[36762]))
        self.stack.enter_context(patch.object(entry.overlap,'_process_name',return_value='python3'))
        self.stack.enter_context(patch.object(entry.original,'_available_ram_bytes',side_effect=lambda:self.available))
        self.stack.enter_context(patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=50*entry.GIB)))

    def fake(self,*,bad=False,late=False,source_failure=False):
        from pt_agent.experiments import heir_common_model_pipeline
        def process(command,**options):
            self.calls.append((command,options));put(Path(options['log']),'synthetic observation only\n');put(Path(options['resources']),'')
            if '--worker'in command:
                out=Path(command[command.index('--out')+1]);put(out/'result.json',dict(status='ok',lowered_adapter_contract=dict(role='fixture')))
                put(out/'model.mlir','synthetic arithmetic source metadata only\n')
            return clean(exit_code=1 if bad else 0)
        def snapshot(out):(out/'source').mkdir();return []
        def native(*args):
            if late:self.available=entry.GIB
            return dict(static_bootstrap_operations=2)
        with (patch.object(heir_common_model_pipeline,'sources',return_value=self.source),
                patch.object(entry.original,'source_gate',side_effect=ValueError('source rejected')if source_failure else None,return_value=self.checked),
                patch.object(entry.original,'_snapshot',side_effect=snapshot),patch.object(entry.original,'run_process',side_effect=process),
                patch.object(entry.original,'check_worker',return_value=dict(passed=True))as check,
                patch.object(entry.original,'check_native',side_effect=native)):
            result=entry.execute(self.args)
        self.math_check=check.call_args;return result

    def test_direct_original_worker_preserves_all_source_flags_and_AS24(self):
        result=self.fake();self.assertEqual(result['status'],'ok',result['error']);self.assertEqual(len(self.calls),2)
        command,options=self.calls[0];self.assertEqual(command[:4],[str(entry.host_python()),'-B','-m',entry.original.MODULE])
        self.assertIn('--worker',command);self.assertNotIn('pt_agent.tools.nixrun',command);self.assertNotIn('--python',command)
        self.assertEqual(command[command.index('--memory-gib')+1],'24')
        for name,path in self.source.items():self.assertEqual(command[command.index('--'+name.replace('_','-'))+1],str(path))
        self.assertEqual(options['mem_gib'],24);self.assertEqual(options['fsize_gib'],16);self.assertLessEqual(options['timeout'],1920)
        self.assertEqual(self.math_check.args[1],self.checked)

    def test_same_native_pipeline_parameters_and_original_numeric_code_unchanged(self):
        namespace=entry._namespace(self.args)
        for name in ('worker','_runtime','run_probes','check_worker','native_command','check_native','serial_lock'):
            self.assertIs(namespace[name],getattr(entry.original,name))
        result=self.fake();native,options=self.calls[1]
        self.assertEqual(native[0],str(self.args.heir_opt));self.assertIn('--mlir-disable-threading',native)
        self.assertEqual(entry.original.read(self.args.out/'native/native-parameters.json'),self.checked['native_parameters'])
        self.assertEqual(options['mem_gib'],24);self.assertEqual(options['fsize_gib'],16);self.assertLessEqual(options['timeout'],7200)
        self.assertTrue(result['strict_math_passed']);self.assertTrue(result['model_compiled_accepted'])

    def test_environment_cleans_inherited_libraries_venvs_and_threads_for_host_worker(self):
        inherited=dict(HOME='/retained-home',LD_LIBRARY_PATH='/bad',LD_PRELOAD='/bad',NIX_PATH='bad',__NIX_X='bad',
            HECATE_PYTHON_LIBRARY_PATH='/bad',PYTHONHOME='/bad',PYTHONPATH='/bad',VIRTUAL_ENV='/bad',CONDA_PREFIX='/bad',OMP_NUM_THREADS='8')
        original=inherited.copy();cleaned=entry.host_environment(inherited)
        self.assertEqual(inherited,original);self.assertEqual(cleaned['HOME'],'/retained-home')
        for name in inherited:
            if name in entry.REMOVE_KEYS or name.startswith(entry.REMOVE_PREFIXES):self.assertNotIn(name,cleaned)
        for name in entry.THREAD_KEYS:self.assertEqual(cleaned[name],'1')
        with patch.dict(os.environ,{'HECATE_PYTHON_LIBRARY_PATH':'foreign','CONDA_PREFIX':'foreign'}):self.fake()
        env=self.calls[0][1]['env'];self.assertNotIn('HECATE_PYTHON_LIBRARY_PATH',env);self.assertNotIn('CONDA_PREFIX',env)

    def test_runtime_scope_registry_and_original_failed_Nix_case_remain_separate(self):
        old=dict(run_id='original-nix-failure',round=entry.base.ROUND,mode=entry.base.MODE,metrics=dict(attempted=dict(export_math=1)))
        put(self.args.registry,old);original=self.args.registry.read_text();result=self.fake()
        self.assertEqual(result['round'],entry.ROUND);self.assertEqual(result['mode'],entry.MODE)
        self.assertTrue(result['runtime_variant'].startswith('host-runtime'))
        self.assertEqual(result['host_runtime']['last_observed_import_versions'],entry.OBSERVED_HOST_VERSIONS)
        self.assertFalse(result['host_runtime']['original_Nix_runtime_equivalence_claimed'])
        self.assertIsNone(result['limits']['nix_seconds']);self.assertEqual(result['limits']['legacy_Nix_seconds_not_applied'],1860)
        lines=self.args.registry.read_text().splitlines();self.assertEqual(lines[0],original.strip());self.assertEqual(len(lines),2)
        row=json.loads(lines[1]);self.assertEqual(len(row),8);self.assertTrue(row['metrics']['model_compiled_accepted'])
        self.assertIn('Torch2.5.1',row['metrics']['scope']);self.assertEqual(row['metrics']['bert_signal_actions'],0)

    def test_changed_interpreter_or_original_result_prefix_rejected_before_run(self):
        self.args.python=Path('/foreign/python')
        with self.assertRaises(ValueError):entry.execute(self.args)
        self.args.python=entry.host_python();self.args.out=self.root/'heir-vgg16-fused-pool-conv-cpu-overlap-old'
        with self.assertRaises(ValueError):entry.execute(self.args)
        self.assertEqual(self.calls,[])

    def test_original_guard_rejects_stale_BERT_foreign_compiler_and_RSS10(self):
        self.args.out.mkdir()
        namespace=entry._namespace(self.args);guard=namespace['physical_observation']
        self.assertFalse(guard(None,12*entry.GIB,50*entry.GIB)['stop'])
        self.assertTrue(guard(dict(subtree_rss_bytes=11*entry.GIB),12*entry.GIB,50*entry.GIB)['stop'])
        self.table[65432]=dict(pid=65432,ppid=1,starttime=11,state='R')
        with patch.object(entry.overlap,'_process_name',side_effect=lambda pid:next(iter(entry.overlap.COMPILERS))if pid==65432 else'python3'):
            self.assertTrue(guard(None,12*entry.GIB,50*entry.GIB)['stop'])
        self.table.pop(65432)
        self.table[64170]['starttime']+=1;self.assertTrue(guard(None,12*entry.GIB,50*entry.GIB)['stop'])

    def test_failed_export_does_not_start_native_and_preserves_observation(self):
        result=self.fake(bad=True);self.assertNotEqual(result['status'],'ok');self.assertEqual(len(self.calls),1)
        self.assertFalse(result['model_compiled_accepted']);self.assertEqual(result['attempted']['native_compile'],0)
        self.assertTrue((self.args.out/'phases/export_math/process-return.json').is_file())
        self.assertTrue((self.args.out/'completion.json').is_file())

    def test_late_parallel_guard_revokes_acceptance_preserving_actual_native_fact(self):
        result=self.fake(late=True);self.assertNotEqual(result['status'],'ok')
        self.assertTrue(result['model_compiled']);self.assertFalse(result['model_compiled_accepted'])
        self.assertEqual(result['completed'],dict(export_math=1,native_compile=1))
        self.assertTrue(result['terminal_parallel_guard']['stop'])

    def test_new_runtime_variant_may_attempt_once_only(self):
        put(self.args.registry,dict(run_id='already-attempted-host',round=entry.ROUND,mode=entry.MODE,metrics=dict(attempted=dict(export_math=1))))
        result=self.fake();self.assertNotEqual(result['status'],'ok');self.assertEqual(self.calls,[])
        self.assertEqual(result['attempted'],dict(export_math=0,native_compile=0))

    def test_source_rejection_is_durable_without_worker_or_model(self):
        result=self.fake(source_failure=True);self.assertEqual(self.calls,[])
        self.assertFalse(result['model_compiled_accepted']);self.assertTrue((self.args.out/'completion.json').is_file())


if __name__=='__main__':unittest.main()
