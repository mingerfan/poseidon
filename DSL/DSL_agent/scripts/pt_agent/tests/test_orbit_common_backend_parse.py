"""Tiny saved-metadata fixtures and fake processes; no Go or model runs."""
from argparse import Namespace
import copy
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import orbit_common_backend_parse as entry


class CommonParserTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False)+'\n')

    @staticmethod
    def clean(**changes):
        value = dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])
        value.update(changes);return value

    def completion(self, case):
        self.write(case/'completion.json', dict(status='ok', required_artifacts_persisted=True, registry_persisted=True))

    def save_source(self, case, result):
        self.write(case/'result.json', result)
        for phase, row in result['phase_results'].items():
            self.write(case/'work'/(phase+'-worker')/'phase_result.json', row)

    def fixture(self, model='vgg16'):
        parser = entry.parser_entry;factory = entry.MODEL_RECORDS[model]['factory']
        orbit = self.root/'orbit';module = orbit/'backend/lattigo';package = module/'lowering/fhe';package.mkdir(parents=True)
        for name in ('main.go','parser.go','scheme.go','exeorder.go'):(package/name).write_text('package main\n')
        sdk = self.root/'sdk';(sdk/'bin').mkdir(parents=True);(sdk/'src').mkdir()
        (sdk/'bin/go').write_text('not an executable Go tool');(sdk/'src/fixture.go').write_text('package fixture\n')
        profile = orbit/'cost_models/profiled_LATTIGONEW_CPU64k_3_16.json';self.write(profile, dict(polynomialDegree=131072))
        front = self.root/'frontend';tracepath = front/'work/trace-worker/trace/model.mlir';tracepath.parent.mkdir(parents=True);tracepath.write_text('metadata fixture only\n')
        constant = front/'work/trace-worker/constants/model.cst';constant.parent.mkdir();constant.write_bytes(b'no numeric payload fixture')
        graph = dict(nodes=[dict(id='arg0',op='input'),dict(id='0',op='constant'),dict(id='1',op='add')],
            edges=[dict(source='arg0',target='1'),dict(source='0',target='1')],inputs=['arg0'],outputs=['1'])
        tdag = front/'work/parse-worker/tdag.json';self.write(tdag,graph)
        base = dict(round=entry.frontend.ROUND,model=factory,status='ok',elapsed_ns=1,
            optimizer_ready=False,backend_ready=False,minimal_ckks_passed=False,formal_matrix_complete=False,
            **dict.fromkeys(entry.frontend.worker.ZERO_FIELDS,0),**dict.fromkeys(entry.frontend.worker.UNMEASURED_FIELDS,None))
        trace = dict(base,phase='trace',source_checked=True,environment_restored=True,orbit_model_trace_passed=True,
            activation_calls=entry.frontend.worker.MODEL_RECORDS[factory]['activations'],
            activation_names=list(range(entry.frontend.worker.MODEL_RECORDS[factory]['activations'])),parse_runs=0,
            mlir=dict(path=str(tracepath),size_bytes=tracepath.stat().st_size),constant_files=[dict(path=str(constant),size_bytes=constant.stat().st_size)])
        for key in ('checkpoint_load_attempts','checkpoint_load_completed','model_build_attempts','model_build_completed',
            'expr_load_attempts','expr_load_completed','function_registration_attempts','function_registrations',
            'save_attempts','save_completed','finalize_attempts','finalize_completed','get_model_calls','trace_runs'):trace[key]=1
        parse = dict(base,phase='parse',tdag_parse_passed=True,trace_runs=0,tdag=str(tdag),
            parser_metadata=dict(poly_deg=131072,max_slot=65536,profile=str(profile)),
            graph_check=entry.frontend.worker._check_graph(graph))
        for key in ('params_calls','params_completed','build_from_mlir_calls','build_from_mlir_completed','parse_runs'):parse[key]=1
        front_result = dict(round=entry.frontend.ROUND,mode=entry.frontend.MODE,model=factory,model_key=model,status='ok',
            orbit_model_trace_passed=True,tdag_parse_passed=True,attempted=dict(trace=1,parse=1),completed=dict(trace=1,parse=1),
            phase_results=dict(trace=trace,parse=parse),processes=[dict(phase=p,observation=self.clean()) for p in ('trace','parse')],
            source_records=dict(model=factory,original_model_modified=False,weights_modified=False,polynomial_modified=False,original_profile=str(profile)))
        self.save_source(front,front_result);self.completion(front)
        case = self.root/'optimizer';mlir = case/'work/optimized.mlir';mlir.parent.mkdir(parents=True);mlir.write_text('optimized metadata fixture\n')
        placed = copy.deepcopy(graph)
        for node in placed['nodes']:node.update(level=3,scale=51)
        placed['nodes'].append(dict(id='2',op='bootstrap',level=16,scale=51,weight=1,op_descr=dict(targetLevel=16)))
        placed['edges'].append(dict(source='1',target='2'));placed['outputs']=['2']
        placedpath=case/'work/validate-worker/tdag.json';self.write(placedpath,placed)
        phases={}
        for phase in ('optimize','validate'):
            row=dict(round=entry.optimizer.ROUND,mode=entry.optimizer.MODE,model=factory,phase=phase,status='ok',elapsed_ns=1,
                source_checked=True,params_checked=True,environment_restored=True,budget_monitor_completed=True,native_optimizer_completed=True,
                params_calls=1,params_completed=1,backend_ready=False,formal_matrix_complete=False,security_validated=False,accuracy_validated=False,
                output_mlir=dict(path=str(mlir),size_bytes=mlir.stat().st_size),
                **dict.fromkeys(entry.optimizer.worker.ZERO_FIELDS,0),**dict.fromkeys(entry.optimizer.worker.UNMEASURED_FIELDS,None))
            row['params']=dict(profile=str(profile),mode='compile' if phase=='optimize' else 'execute',poly_deg=131072,max_slot=65536,Sf=51,Sw=40,Csw=40,
                lvl_lb=1,lvl_ub=16,bts_lb=3,bts_ub=16,bpsdepth=15,threads=1,comp=True,part=True,reqbp=False,
                netname=entry.frontend.worker.MODEL_RECORDS[factory]['benchmark'],backend='Lattigo',trunc_val=1,dacapo_mlir_in=True,dacapo_mlir_out=True)
            if phase=='optimize':row.update(optimize_attempts=1,optimize_completed=1,seed_calls=1,seed_completed=1,observer_restored=True,
                solve_ilp_errors=0,source_record=str(front/'result.json'))
            else:row.update(validate_attempts=1,validate_completed=1,build_from_mlir_calls=1,build_from_mlir_completed=1,
                check_tdag_calls=1,check_tdag_completed=1,placement_artifact_checked=True,tdag=str(placedpath),
                graph_check=entry.optimizer.worker._placement_record(placed),source_record=str(case/'work/optimize-worker/phase_result.json'))
            phases[phase]=row
        result=dict(round=entry.optimizer.ROUND,mode=entry.optimizer.MODE,model=factory,model_key=model,status='ok',
            attempted=dict(optimize=1,validate=1),completed=dict(optimize=1,validate=1),phase_results=phases,
            native_optimizer_completed=True,placement_artifact_checked=True,
            processes=[dict(phase=p,observation=self.clean()) for p in phases],
            source_records=dict(source_record=str(front/'result.json'),input_mlir=dict(trace['mlir']),
                original_graph_nodes=3,original_graph_edges=2))
        self.save_source(case,result);self.completion(case)
        build=self.root/'shared';cache=build/'work/module-cache';cache.mkdir(parents=True);(cache/'fixture.go').write_text('package dependency\n')
        (build/'orbit-fhe').write_bytes(b'shared interpreter fixture')
        shared=dict(round=parser.shared.ROUND,mode=parser.shared.MODE,status='ok',runtime_interpreter_compiled=True,
            go_build_attempts=1,go_build_completed=1,cleanup_unknown=False,process=self.clean(),
            source=dict(model=parser.shared.MODEL),package=dict(interpreter_package=str(package)),
            binary=dict(path=str(build/'orbit-fhe'),bytes=(build/'orbit-fhe').stat().st_size),**dict.fromkeys(parser.shared.ZERO_FIELDS,0))
        self.write(build/'result.json',shared);self.write(build/'build-argv.json',parser.shared.build_command(sdk,build/'orbit-fhe'));self.completion(build)
        audit=self.root/'audit.json';audited=[]
        for scope,root,relative in (('module',module,'lowering/fhe/main.go'),('sdk',sdk,'src/fixture.go'),('cache',cache,'fixture.go')):
            st=(root/relative).stat();audited.append(dict(scope=scope,relative=relative,bytes=st.st_size,mtime_ns=st.st_mtime_ns))
        self.write(audit,dict(status='passed_source_no_model_context_keys_or_payload_initializers',unresolved_imports=[],
            roots=dict(module=str(module),sdk=str(sdk),cache=str(cache)),audited_files=audited,
            explicit_review=dict(model_context_initializers=0,keygen_initializers=0,model_arithmetic_initializers=0,external_weight_or_constant_payload_reads=0)))
        return Namespace(model=model,out=self.root/'new-case',source_case=case,shared_build_case=build,orbit_root=orbit,
            go_sdk=sdk,initialization_audit=audit,registry=self.root/'registry.jsonl',run=False)

    def test_complete_common_sources_accept_each_target_independent_of_shared_res_model(self):
        for model in ('vgg16','alexnet'):
            with self.subTest(model=model), TemporaryDirectory(dir=self.root) as directory:
                original=self.root;self.root=Path(directory)
                try:
                    args=self.fixture(model);checked=entry.source_gate(args)
                    self.assertEqual(checked['target_model'],entry.MODEL_RECORDS[model]['factory'])
                    self.assertEqual(checked['shared_build_model'],entry.parser_entry.shared.MODEL)
                    self.assertFalse(checked['shared_build_proves_target_model'])
                    self.assertEqual(checked['expected_graph']['outputs'],['2'])
                finally:self.root=original

    def test_gate_never_opens_mlir_or_external_constant_payload(self):
        args=self.fixture();original=Path.read_text
        def guarded(path,*a,**kw):
            if path.suffix in ('.mlir','.cst'):raise AssertionError('payload opened')
            return original(path,*a,**kw)
        with patch.object(Path,'read_text',guarded):self.assertEqual(entry.source_gate(args)['expected_graph']['node_count'],4)

    def test_wrong_target_failed_phase_and_incomplete_source_are_rejected(self):
        args=self.fixture();original=entry.read(args.source_case/'result.json')
        for kind in ('model','status','attempt','cleanup','source'):
            changed=copy.deepcopy(original)
            if kind=='model':changed['model']=entry.MODEL_RECORDS['alexnet']['factory']
            elif kind=='status':changed['status']='error'
            elif kind=='attempt':changed['attempted']['optimize']=True
            elif kind=='cleanup':changed['processes'][0]['observation']['remaining_pids']=[8]
            else:changed['source_records']['source_record']=str(self.root/'foreign/result.json')
            self.write(args.source_case/'result.json',changed)
            with self.subTest(kind=kind),self.assertRaises((ValueError,FileNotFoundError)):entry.source_gate(args)

    def test_saved_worker_source_drift_is_rejected(self):
        args=self.fixture();path=args.source_case/'work/validate-worker/phase_result.json';row=entry.read(path);row['model']='foreign';self.write(path,row)
        with self.assertRaisesRegex(ValueError,'worker and outer'):entry.source_gate(args)

    def test_profile_graph_and_original_frontend_provenance_cannot_be_swapped(self):
        args=self.fixture();original=entry.read(args.source_case/'result.json')
        for kind in ('profile','graph','frontend'):
            result=copy.deepcopy(original)
            if kind=='profile':result['phase_results']['validate']['params']['max_slot']=32768
            elif kind=='graph':result['phase_results']['validate']['graph_check']['outputs']=['1']
            else:result['phase_results']['optimize']['source_record']=str(self.root/'foreign/result.json')
            self.save_source(args.source_case,result)
            with self.subTest(kind=kind),self.assertRaises(ValueError):entry.source_gate(args)

    def test_mutated_frontend_model_weights_or_poly_metadata_is_rejected(self):
        args=self.fixture();path=Path(entry.read(args.source_case/'result.json')['source_records']['source_record']);original=entry.read(path)
        for field in ('model','original_model_modified','weights_modified','polynomial_modified'):
            row=copy.deepcopy(original)
            if field=='model':row['model']='foreign'
            else:row['source_records'][field]=True
            self.save_source(path.parent,row)
            with self.subTest(field=field),self.assertRaises(ValueError):entry.source_gate(args)

    def test_shared_build_module_sdk_and_execution_drift_is_rejected(self):
        args=self.fixture();path=args.shared_build_case/'result.json';original=entry.read(path)
        for kind in ('module','key','signal','unknown','binary'):
            row=copy.deepcopy(original)
            if kind=='module':row['package']['interpreter_package']=str(self.root/'foreign')
            elif kind=='key':row['keygen_runs']=1
            elif kind=='signal':row['process']['signals_sent']=[dict(signal=15)]
            elif kind=='unknown':row['cleanup_unknown']=True
            else:row['binary']['bytes']+=1
            self.write(path,row)
            with self.subTest(kind=kind),self.assertRaises(ValueError):entry.source_gate(args)
        self.write(path,original);self.write(args.shared_build_case/'build-argv.json',['foreign-sdk','build'])
        with self.assertRaisesRegex(ValueError,'SDK'):entry.source_gate(args)

    def test_initialization_audit_remains_mandatory(self):
        args=self.fixture();row=entry.read(args.initialization_audit);row['explicit_review']['keygen_initializers']=1;self.write(args.initialization_audit,row)
        with self.assertRaisesRegex(ValueError,'initialization'):entry.source_gate(args)

    def test_public_go_client_and_original_context_managers_are_unchanged(self):
        args=self.fixture();namespace=entry._namespace(args)
        self.assertIs(entry.CLIENT,entry.parser_entry.CLIENT)
        self.assertIs(namespace['serial_lock'],entry.parser_entry.serial_lock)
        self.assertIs(namespace['check_parsed'],entry.parser_entry.check_parsed)
        self.assertIsNot(namespace['execute'].__globals__,entry.parser_entry.execute.__globals__)
        self.assertIs(entry.parser_entry.execute.__globals__['source_gate'],entry.parser_entry.source_gate)
        with self.assertRaises(TypeError):entry._clone_plain(entry.parser_entry.serial_lock,{})

    @staticmethod
    def parsed(checked):
        source=checked['optimized_source'];row=dict(status='ok',mlir_path=source['optimized_mlir']['path'],mlir_bytes=source['optimized_mlir']['bytes'],
            mlir_text_reads=2,parsed_nodes=4,parsed_operations=3,input_count=1,output_id=2,edge_count=3,operand_edges_with_duplicates=3,
            op_counts=dict(input=1,constant=1,add=1,bootstrap=1),static_bootstrap_nodes=1,topological_order_count=3,
            nodes=[dict(id=i) for i in range(3)],official_ReadFile_calls=1,official_parseMLIROperation_calls=3,
            official_getOptimalRunOrder_calls=1,official_getMaxWorkingPoolSize_calls=1,**dict.fromkeys(entry.ZERO_FIELDS[:9],0))
        return row

    def outer(self, failure=None,run=True):
        args=self.fixture();args.run=run;calls=[];checked=entry.source_gate(args)
        def process(command,**options):
            calls.append((command,options));Path(options['log']).write_text('fake process only\n');Path(options['resources']).write_text('')
            if command[1]=='build':(args.out/'orbit-parser').write_bytes(b'fake parser');return self.clean(exit_code=1 if failure=='build' else 0)
            target=Path(command[command.index('--out')+1])
            if '--self-test' in command:row=dict(status='ok',selftests=6,model_parses=0,**dict.fromkeys(entry.ZERO_FIELDS[:9],0))
            else:
                row=self.parsed(checked)
                if failure=='graph':row['edge_count']=2
            self.write(target,row);return self.clean(timed_out=failure=='timeout' and '--mlir' in command)
        base=entry.parser_entry
        with ExitStack() as stack:
            stack.enter_context(patch.object(base.paths,'RESULTS_DIR',self.root))
            stack.enter_context(patch.object(base,'_available_ram_bytes',return_value=16*base.GIB))
            stack.enter_context(patch.object(base.shutil,'disk_usage',return_value=SimpleNamespace(free=100*base.GIB)))
            stack.enter_context(patch.object(base,'run_process',side_effect=process))
            result=entry.execute(args)
        return args,result,calls

    def test_preparation_has_no_go_launch_and_consumption_stays_unknown(self):
        args,result,calls=self.outer(run=False)
        self.assertEqual(calls,[]);self.assertEqual(result['status'],'prepared');self.assertIsNone(result['backend_parser_consumption_verified'])
        self.assertTrue((args.out/'source/underlying-parser-entry.py').is_file())
        self.assertEqual((args.out/'source/parser-main.go').read_text(),entry.CLIENT)
        self.assertEqual((args.out/'work/telemetry/mode').read_text(),'off\n')

    def test_fake_pipeline_records_actual_target_with_original_finite_budgets(self):
        args,result,calls=self.outer();self.assertEqual(result['status'],'ok');self.assertTrue(result['backend_parser_consumption_verified'])
        self.assertEqual(result['model'],entry.MODEL_RECORDS['vgg16']['factory'])
        self.assertEqual(result['completed'],dict(build=1,selftest=1,parse=1))
        for (_,options),seconds in zip(calls,(600,120,300)):
            self.assertEqual(options['mem_gib'],12);self.assertEqual(options['fsize_gib'],8);self.assertLessEqual(options['timeout'],seconds)
        registry=[json.loads(line) for line in args.registry.read_text().splitlines()]
        self.assertEqual(len(registry),1);self.assertEqual(registry[0]['model'],result['model']);self.assertEqual(registry[0]['round'],entry.ROUND)
        self.assertEqual(set(registry[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        completion=entry.read(args.out/'completion.json');self.assertTrue(completion['registry_persisted']);self.assertTrue(completion['required_artifacts_persisted'])
        self.assertTrue(all(result[key]==0 for key in entry.ZERO_FIELDS));self.assertIsNone(result['security_acceptance'])

    def test_first_build_failure_stops_and_preserves_terminal_record(self):
        args,result,calls=self.outer(failure='build');self.assertEqual(len(calls),1);self.assertEqual(result['status'],'error')
        self.assertEqual(result['attempted'],dict(build=1,selftest=0,parse=0));self.assertTrue((args.out/'completion.json').is_file())

    def test_graph_mismatch_or_timeout_preserves_partial_without_success_claim(self):
        for failure in ('graph','timeout'):
            with self.subTest(failure=failure),TemporaryDirectory(dir=self.root) as directory:
                previous=self.root;self.root=Path(directory)
                try:
                    args,result,calls=self.outer(failure=failure)
                    self.assertEqual(len(calls),3);self.assertIsNone(result['backend_parser_consumption_verified'])
                    self.assertEqual(result['status'],'timeout' if failure=='timeout' else 'error')
                    self.assertTrue((args.out/'parsed-model.json').is_file())
                finally:self.root=previous


if __name__=='__main__':unittest.main()
