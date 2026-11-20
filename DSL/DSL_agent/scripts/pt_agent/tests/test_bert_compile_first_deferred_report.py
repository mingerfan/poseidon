"""Compile-first synthetic metadata, arithmetic and finite orchestration only."""
import ast
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
import numpy as np
from pt_agent.experiments import bert_compile_first_deferred_report as entry
from pt_agent.tests import test_bert_serial_continuation as fixtures


def fixture(root):
    args=fixtures.fixture(root);fixtures.frozen(args.source_case)
    source=dict(model=str(root/'model'),model_info=str(args.source_case/'model/model_info.json'))
    fixtures.checkpoint(args,source)
    args.checkpoint_audit=args.checkpoint_out/'result.json';args.remaining_seconds=11160;args.work_dir=root/'math-work'
    # Fixed synthetic calibration storage. Labels are deliberately unused.
    values=(np.arange(1000*40*4,dtype=np.float64).reshape(1000,40,4)%17-8.)/16.
    np.savez(args.source_case/'model/samples/calibration.npz',inputs_embeds=values,labels=np.zeros(1000,dtype=np.int64))
    return args


def proof(args,source=None):
    source=source or entry.metadata_source(args.source_case,args.checkpoint_audit)
    pairs=[('graph_reference','tensor_exact'),('graph_reference','mock_exact'),('tensor_program','mock_program')]
    return dict(round=entry.ROUND,model=entry.MODEL,mode=entry.MODE,phase='math',status='ok',math_passed=True,
        source=source,source_recheck=copy.deepcopy(source),probe_source=source['calibration'],
        compiler_inputs=entry.compiler_metadata(args),attempted=dict.fromkeys(entry.COUNTERS,3),
        completed=dict.fromkeys(entry.COUNTERS,3),strict_checks_attempted=9,strict_checks_completed=9,calibration_probes=3,
        frozen_math=[dict(sample=i,sample_index=source['calibration']['sample_indices'][i],checks=[
            dict(reference=a,observed=b,passed=True,criterion=entry.original.CRITERION,max_absolute_error=0.,max_tolerance_ratio=0.)
            for a,b in pairs]) for i in range(3)],**dict.fromkeys(entry.ZERO,0))


def numeric_api(log,*,bad=None):
    from pt_agent.experiments.bert_sst2_worker import strict_math
    graph=SimpleNamespace(inputs=['inputs_embeds'],outputs=['logits'],weights={'one':np.array([1.],dtype=np.float64)},
        tensors={'inputs_embeds':SimpleNamespace(shape=(1,40,4))})
    info=SimpleNamespace(graph=graph)
    program=SimpleNamespace(params=[SimpleNamespace(name='inputs_embeds')])
    def formula(values):
        value=values['inputs_embeds']
        return np.stack((value[:,0,0]+value[:,1,1],value[:,2,2]-value[:,3,3]),axis=1)
    def reference(graph,values):log.append('graph');return dict(logits=formula(values))
    def execute(program,weights,values,*,engine,exact):
        log.append((engine,exact));out=formula(values)
        if not exact:out=out+.125
        if bad=='difference' and engine=='mock' and exact:out=out+.001
        if bad=='dtype' and engine=='tensor' and exact:out=out.astype(np.float32)
        return out
    return dict(np=np,load=lambda path:info,parse=lambda text:program,interface=lambda *a:None,
        check=lambda *a,**kw:SimpleNamespace(ok=True),reference=reference,execute=execute,strict=strict_math)


class DeferredReportGateChecks(unittest.TestCase):
    def test_c2_checkpoint_split_required_without_report_access(self):
        with TemporaryDirectory() as tmp:
            args=fixture(Path(tmp));original=entry.serial.read
            def guarded(path):
                self.assertNotEqual(Path(path).name,'REPORT_ACCESS.json')
                return original(path)
            with patch.object(entry.serial,'read',side_effect=guarded):
                source=entry.metadata_source(args.source_case,args.checkpoint_audit)
            self.assertTrue(source['frozen']['frozen']);self.assertTrue(source['semantic_checkpoint_accepted'])
            self.assertTrue(source['report_deferred']);self.assertIsNone(source['report_task_acceptance'])
            self.assertEqual(source['calibration']['sample_indices'],[0,1,2])
            self.assertFalse((args.source_case/'session/REPORT_ACCESS.json').exists())

    def test_unfinished_failed_c2_and_foreign_checkpoint_rejected(self):
        for change in ('unfinished','domain','checkpoint'):
            with self.subTest(change=change),TemporaryDirectory() as tmp:
                args=fixture(Path(tmp))
                if change=='unfinished':
                    path=args.source_case/'session/SESSION.json';record=entry.read(path);record['phase']='tune';fixtures.save(path,record)
                elif change=='domain':
                    path=args.source_case/'session/AGENT.json';record=entry.read(path);record['confirm']['evals'][0]['max_u_calibrated']=.91;fixtures.save(path,record)
                    marker=args.source_case/'session/FROZEN.json';frozen=entry.read(marker);frozen['confirm']=record['confirm'];fixtures.save(marker,frozen)
                else:
                    record=entry.read(args.checkpoint_audit);record['worker']['source']['model']='/foreign';fixtures.save(args.checkpoint_audit,record)
                with self.assertRaises(ValueError):entry.metadata_source(args.source_case,args.checkpoint_audit)

    def test_actual_math_proof_exact_counts_roles_and_source_required(self):
        for change in ('valid','source','partial','wrongpair','nonfinite'):
            with self.subTest(change=change),TemporaryDirectory() as tmp:
                args=fixture(Path(tmp));record=proof(args);path=Path(tmp)/'math.json'
                if change=='source':record['source']['selected_source']='foreign'
                if change=='partial':record['completed']['mock_program']=2
                if change=='wrongpair':record['frozen_math'][0]['checks'][2]['observed']='tensor_exact'
                if change=='nonfinite':record['frozen_math'][0]['checks'][0]['max_tolerance_ratio']=float('nan')
                fixtures.save(path,record)
                if change=='valid':
                    accepted=entry.compile_source(args.source_case,args.checkpoint_audit,path)
                    self.assertIsNone(accepted['report_result']);self.assertEqual(accepted['mathematical_groups_passed'],3)
                else:
                    with self.assertRaises(ValueError):entry.compile_source(args.source_case,args.checkpoint_audit,path)

    def test_incomplete_report_bytes_never_opened_or_changed(self):
        with TemporaryDirectory() as tmp:
            args=fixture(Path(tmp));marker=args.source_case/'session/REPORT_ACCESS.json'
            marker.write_text('retained incomplete or malformed marker, deliberately never parsed')
            original=Path.read_text
            def reader(path,*a,**kw):
                if path==marker:raise AssertionError('report marker forbidden')
                return original(path,*a,**kw)
            with patch.object(Path,'read_text',reader):
                source=entry.metadata_source(args.source_case,args.checkpoint_audit)
                record=proof(args,source);path=Path(tmp)/'math.json';fixtures.save(path,record)
                self.assertTrue(entry.compile_source(args.source_case,args.checkpoint_audit,path)['report_deferred'])
            self.assertEqual(marker.read_text(),'retained incomplete or malformed marker, deliberately never parsed')


class CalibrationMathChecks(unittest.TestCase):
    def test_original_three_groups_on_only_saved_first3_and_no_labels(self):
        with TemporaryDirectory() as tmp:
            args=fixture(Path(tmp));calls=[];load=np.load;access=[]
            class GuardArchive:
                def __init__(self,path,**kw):
                    self.archive=load(path,**kw);access.append(str(path))
                    self.path=path
                def __enter__(self):return self
                def __exit__(self,*exc):self.archive.close()
                def __getitem__(self,key):
                    if key!='inputs_embeds':raise AssertionError('only model inputs permitted')
                    return self.archive[key]
            with (patch.object(entry,'_math_api',return_value=numeric_api(calls)),
                    patch.object(entry.original,'native_metadata_commands',lambda template:nullcontext()),
                    patch.object(np,'load',GuardArchive)):
                result=entry.math_worker(args)
            self.assertEqual(result['status'],'ok',result['error']);self.assertTrue(result['math_passed'])
            self.assertEqual(result['completed'],dict.fromkeys(entry.COUNTERS,3))
            self.assertEqual(result['strict_checks_completed'],9);self.assertEqual(len(calls),15)
            self.assertEqual(access,[str(args.source_case/'model/samples/calibration.npz')])
            self.assertEqual(result['input_metadata']['inputs_embeds']['shape'],[3,40,4])
            self.assertFalse((args.source_case/'session/REPORT_ACCESS.json').exists())
            self.assertTrue(entry.compile_source(args.source_case,args.checkpoint_audit,args.work_dir/'math-result.json')['report_deferred'])

    def test_math_error_stops_first_sample_and_persists_partial(self):
        for bad in ('difference','dtype'):
            with self.subTest(bad=bad),TemporaryDirectory() as tmp:
                args=fixture(Path(tmp));calls=[]
                with patch.object(entry,'_math_api',return_value=numeric_api(calls,bad=bad)), \
                        patch.object(entry.original,'native_metadata_commands',lambda template:nullcontext()):result=entry.math_worker(args)
                self.assertEqual(result['status'],'error');self.assertFalse(result['math_passed'])
                self.assertEqual(len(calls),5);self.assertEqual(result['completed'],dict.fromkeys(entry.COUNTERS,1))
                self.assertLess(result['strict_checks_completed'],3)
                self.assertEqual(entry.read(args.work_dir/'math-result.json')['status'],'error')

    def test_bad_calibration_shape_rejected_before_forward(self):
        with TemporaryDirectory() as tmp:
            args=fixture(Path(tmp));np.savez(args.source_case/'model/samples/calibration.npz',inputs_embeds=np.zeros((2,40,4)))
            calls=[]
            with patch.object(entry,'_math_api',return_value=numeric_api(calls)), \
                    patch.object(entry.original,'native_metadata_commands',lambda template:nullcontext()):result=entry.math_worker(args)
            self.assertEqual(result['status'],'error');self.assertEqual(calls,[])
            self.assertEqual(result['strict_checks_attempted'],0)


class PrivateCompilerAndOuterChecks(unittest.TestCase):
    def test_private_two_argument_seams_and_original_math_bodies_unchanged(self):
        old_emit=entry.original._emit;old_command=entry.original.phase_command
        namespace=entry.compile_namespace(10000)
        self.assertIsNot(namespace['execute'],entry.original.execute)
        self.assertIs(namespace['_emit'].__code__,old_emit.__code__)
        self.assertIs(namespace['_child'].__code__,entry.original._child.__code__)
        self.assertIs(entry.original.phase_command,old_command)
        self.assertNotIn('main',namespace)
        self.assertEqual(entry.original.CASE_BYTES,8*entry.GIB);self.assertEqual(namespace['CASE_BYTES'],32*entry.GIB)
        self.assertEqual(namespace['MODE'],entry.MODE);self.assertEqual(namespace['OVERALL_SECONDS'],10000)
        for name in ('worker','execute'):
            tree=entry._math_argument(getattr(entry.original,name))
            attrs=[node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='args']
            self.assertIn('math_result',attrs);self.assertNotIn('report_result',attrs)
        with patch.object(entry,'run_process',return_value=fixtures.observed()) as child:
            namespace['run_process'](['fake'],mem_gib=12,fsize_gib=8)
        self.assertEqual(child.call_args.kwargs,dict(mem_gib=24,fsize_gib=16))
        for value in (True,0,12001,float('nan')):
            with self.assertRaises(ValueError):entry.compile_namespace(value)

    def test_compiler_worker_argv_has_real_math_result_no_report(self):
        args=SimpleNamespace(intake_python=Path('/intake'),**{key:Path('/source')/key for key in
            ('source_case','checkpoint_audit','math_result','metadata_template','profile','frontend','model_info','program')})
        argv=entry.compile_namespace()['phase_command'](args,'compile',Path('/isolated'))
        self.assertEqual(argv[argv.index('--seconds')+1],'7260')
        self.assertIn(entry.MODULE,argv);self.assertIn('--math-result',argv);self.assertNotIn('--report-result',argv)
        self.assertEqual(argv[argv.index('--math-result')+1],str(args.math_result))

    def run_outer(self,root,*,fail=None,signal_number=None):
        args=fixture(root);source=entry.metadata_source(args.source_case,args.checkpoint_audit)
        def child(argv,**kwargs):
            phase='math' if '--worker-phase' in argv else 'compile'
            self.assertEqual(kwargs['mem_gib'],24);self.assertEqual(kwargs['fsize_gib'],16)
            self.assertLessEqual(kwargs['timeout'],entry.OVERALL_SECONDS)
            if signal_number:return fixtures.observed(exit_code=-signal_number,signal=signal_number)
            if phase==fail:return fixtures.observed(exit_code=1)
            path=args.out/'math/math-result.json'
            if phase=='math':fixtures.save(path,proof(args,source))
            else:
                accepted=entry.compile_source(args.source_case,args.checkpoint_audit,path)
                fixtures.save(args.compile_out/'result.json',dict(round=entry.ROUND,model=entry.MODEL,mode=entry.MODE,status='ok',source=accepted,
                    completed=dict(emit=1,trace=1,compile=1),compiled_bootstraps=9,runtime_plan='/synthetic-plan'))
                fixtures.save(args.compile_out/'completion.json',dict(status='ok',registry_persisted=True,required_artifacts_persisted=True))
            return fixtures.observed()
        with (patch.object(entry.paths,'RESULTS_DIR',args.out.parent),
                patch.object(entry.serial,'foreign_heavy_processes',return_value=[]),
                patch.object(entry,'run_process',side_effect=child) as children,
                patch.object(entry.original,'_available_ram_bytes',return_value=50*entry.GIB),
                patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=50*entry.GIB)),
                patch('pt_agent.registry.git_rev',return_value='fixture')):result=entry.execute(args)
        return args,result,children

    def test_fake_full_chain_has_two_phases_zero_reports_and_unique_registry(self):
        with TemporaryDirectory() as tmp:
            args,result,children=self.run_outer(Path(tmp))
            self.assertEqual(result['status'],'ok',result['error']);self.assertEqual(children.call_count,2)
            self.assertEqual(result['completed'],dict(math=1,compile=1));self.assertTrue(result['compile_completed'])
            self.assertEqual(result['compiled_bootstraps'],9);self.assertTrue(result['report_deferred'])
            self.assertIsNone(result['report_task_acceptance']);self.assertIsNone(result['accuracy_acceptance'])
            self.assertFalse((args.source_case/'session/REPORT_ACCESS.json').exists())
            for key in entry.ZERO:self.assertEqual(result[key],0)
            rows=[json.loads(line) for line in (args.out.parent/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(set(rows[0]),entry.serial.REGISTRY_FIELDS)
            completion=entry.read(args.out/'completion.json');self.assertEqual(completion['case_artifact_bytes'],entry.artifact_bytes(args.out))
            self.assertTrue(completion['registry_persisted']);self.assertEqual(completion['status'],'ok')

    def test_first_error_or_signal_stops_and_preserves_partial(self):
        for fail in ('math','compile','signal'):
            with self.subTest(fail=fail),TemporaryDirectory() as tmp:
                args,result,children=self.run_outer(Path(tmp),fail=fail if fail!='signal' else None,signal_number=15 if fail=='signal' else None)
                self.assertEqual(result['status'],'error');self.assertEqual(children.call_count,2 if fail=='compile' else 1)
                self.assertEqual(result['error_stage'],'compile' if fail=='compile' else 'math')
                self.assertTrue((args.out/'completion.json').is_file())
                if fail=='signal':self.assertEqual(result['termination'],dict(cause='incoming_signal',signal=15))

    def test_foreign_heavy_or_unfrozen_prevents_child(self):
        with TemporaryDirectory() as tmp:
            args=fixture(Path(tmp));state=entry.read(args.source_case/'session/SESSION.json');state['phase']='tune'
            fixtures.save(args.source_case/'session/SESSION.json',state)
            with (patch.object(entry.paths,'RESULTS_DIR',args.out.parent),
                    patch.object(entry,'run_process',side_effect=AssertionError('no child')) as children,
                    patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=50*entry.GIB)),
                    patch('pt_agent.registry.git_rev',return_value='fixture')):result=entry.execute(args)
            self.assertEqual(result['status'],'error');self.assertEqual(result['error_stage'],'admission');children.assert_not_called()

    def test_parser_no_report_input_or_inconsistent_modes(self):
        common=['--source-case','/source','--checkpoint-audit','/audit/result.json','--metadata-template','/saved/argv.json']
        for extra in (['--report-result','/old/report'],['--worker-phase','math','--delegate-compile']):
            with patch('sys.stderr'),self.assertRaises(SystemExit):entry.parser().parse_args(common+extra)

    def test_delegate_main_uses_real_math_argument_and_rejects_parameter_drift(self):
        with TemporaryDirectory() as tmp:
            args=fixture(Path(tmp));path=Path(tmp)/'accepted-math.json';record=proof(args);fixtures.save(path,record)
            argv=['--delegate-compile','--out',str(args.compile_out),'--math-result',str(path)]
            for name in ('source_case','checkpoint_audit','metadata_template','profile','frontend','intake_python'):
                argv+=['--'+name.replace('_','-'),str(getattr(args,name))]
            with patch.object(entry,'compile_namespace',return_value={'execute':lambda actual:dict(status='ok',error=None)}) as namespace,patch('builtins.print'):
                self.assertEqual(entry.main(argv),0)
            namespace.assert_called_once_with(11160)
            record['compiler_inputs']['profile_metadata']['foreign_parameter']=1;fixtures.save(path,record)
            with patch.object(entry,'compile_namespace',side_effect=AssertionError('no compiler delegate')):
                with self.assertRaisesRegex(ValueError,'parameters/metadata'):entry.main(argv)

    def test_shared_case_and_disk_reserve_are_bounded_before_another_phase(self):
        args=SimpleNamespace(out=Path('/fixture-outer'),compile_out=Path('/fixture-compiler'))
        with patch.object(entry,'_case_size',return_value=entry.CASE_BYTES-entry.RESERVE), \
                patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=10*entry.GIB)):
            entry._budget(args)
        with patch.object(entry,'_case_size',return_value=entry.CASE_BYTES-entry.RESERVE+1), \
                patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=10*entry.GIB)):
            with self.assertRaises(RuntimeError):entry._budget(args)
        with patch.object(entry,'_case_size',return_value=0), \
                patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=10*entry.GIB-1)):
            with self.assertRaises(RuntimeError):entry._budget(args)


if __name__=='__main__':unittest.main()
