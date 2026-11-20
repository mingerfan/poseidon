"""Bounded fake layout-source/Go-compile provenance and dispatch checks."""
import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_layout_16384_go_compile as entry
from pt_agent.tests import test_heir_no_fusion_go_compile as fixtures


class Layout16384GoCases(unittest.TestCase):
    persist=fixtures.NoFusionGoCases.persist
    inherited_gate=fixtures.NoFusionGoCases.inherited_gate
    fake_process=fixtures.NoFusionGoCases.fake_process

    def setUp(self):
        fixtures.NoFusionGoCases.setUp(self)
        self.new_gate_patch=patch.object(entry.native,'source_gate',side_effect=self.inherited_gate)
        self.new_gate_patch.start();self.addCleanup(self.new_gate_patch.stop)
        self.new_scan_patch=patch.object(entry.native,'_scan',return_value={})
        self.new_scan_patch.start();self.addCleanup(self.new_scan_patch.stop)

    def configure(self,model,variant):
        fixtures.NoFusionGoCases.configure(self,model,variant)
        self.inherited['candidate_configuration']=dict(original_linear_fusion_retained=True,
            pass_added=entry.layout.LAYOUT_PASS,added_count=1,immediately_after=entry.layout.PROPAGATION_PASS,
            other_pass_order_unchanged=True,original_failure_preserved=True,prior_default_layout_crash_risk_retained=True)
        directory=self.case/'native'
        pipeline=entry.layout.add_layout_optimization(entry.native._pipeline_at(self.inherited['original_source'],directory))
        argv=entry.native._compiler_argv(str(self.tool),Path(self.inherited['original_source_mlir']['path']),directory,pipeline)+list(entry.native.FAILURE_FLAGS)
        self.record.update(round=entry.layout.ROUND,mode=entry.layout.MODE,variant=entry.layout.VARIANT,
            source=copy.deepcopy(self.inherited),original_linear_fusion_retained=True,
            prior_default_layout_crash_risk_retained=True,native_argv=argv)
        self.record['phases']['native_compile']['argv']=argv
        request=self.record['native_request'];request.pop('pass_removed');request.pop('removed_count')
        request.update(pass_added=entry.layout.LAYOUT_PASS,added_count=1)
        self.completion['round']=entry.layout.ROUND
        fixtures.put(self.case/'source.json',self.record['source'])
        fixtures.put(directory/'explicit.pipeline',pipeline+'\n');self.persist()

    def execute_entry(self,process=None):
        with patch.object(entry.paths,'RESULTS_DIR',self.root), \
                patch.object(entry.common,'run_process',side_effect=process or self.fake_process), \
                patch.object(entry.common,'_available_ram_bytes',return_value=3*entry.native.GIB), \
                patch.object(entry.common.shutil,'disk_usage',return_value=SimpleNamespace(free=20*entry.native.GIB)):
            return entry.execute(self.args)

    def test_both_explicit_layout_sources_retain_original_math_and_native_clocks(self):
        for model,variant,pairs in [('alexnet','alex-serial-json',21),('vgg16','vgg-rss12-json-overlap',45)]:
            self.configure(model,variant)
            checked=entry.model_source_gate(self.args);lineage=checked['layout_16384_source']
            self.assertEqual(checked['strict_math']['activation_pairs'],pairs)
            self.assertEqual(checked['strict_math']['training_samples'],3)
            self.assertTrue(lineage['original_linear_fusion_retained'])
            self.assertEqual(lineage['pass_added'],entry.layout.LAYOUT_PASS)
            self.assertEqual(lineage['source_native_process_seconds'],1e-7)
            self.assertEqual(lineage['complete_case_seconds'],1.5)
            self.assertFalse(lineage['model_specific_native_codegen_verified'])
            self.assertFalse(checked['previous_lowering_clock_included'])
            self.assertEqual(checked['input_mlir']['path'],str(self.case/'native/model-lattigo.mlir'))

    def test_gate_semantic_AST_only_retargets_pass_and_lineage_sites(self):
        tree=entry._gate_tree()
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant)and node.value in('pass_added','added_count',entry.layout.LAYOUT_PASS):
                node.value={'pass_added':'pass_removed','added_count':'removed_count',entry.layout.LAYOUT_PASS:'linalg-fuse-linear-ops'}[node.value]
            if isinstance(node,ast.Attribute)and isinstance(node.value,ast.Name)and node.value.id=='native'and node.attr=='add_layout_optimization':
                node.attr='remove_linear_fusion'
            if isinstance(node,ast.keyword)and node.arg=='layout_16384_source':node.arg='no_fusion_source'
        self.assertEqual(ast.dump(tree,include_attributes=False),ast.dump(ast.parse(inspect.getsource(entry.base.model_source_gate)),include_attributes=False))

    def test_old_host_and_no_fusion_source_gates_still_reject_layout_identity(self):
        with self.assertRaises(ValueError):entry.host_sources.model_source_gate(self.args)
        with self.assertRaises(ValueError):entry.base.model_source_gate(self.args)

    def test_unaccepted_monitor_pending_wrong_identity_or_math_rejects(self):
        for key,value in [('status','prepared'),('model_compiled_accepted',False),('monitor_stopped',False),
                ('round',entry.base.native.ROUND),('variant',entry.base.native.VARIANT),('math_inherited',False),
                ('new_math_measured',True),('original_linear_fusion_retained',False),('child_timeout_signal',15)]:
            prior=self.record[key];self.record[key]=value;self.persist()
            with self.subTest(key=key),self.assertRaises(ValueError):entry.model_source_gate(self.args)
            self.record[key]=prior

    def test_pipeline_complete_argv_requested_parameters_and_output_required(self):
        original=copy.deepcopy(self.record)
        mutations=[lambda:self.record['native_request'].update(pass_removed='linalg-fuse-linear-ops'),
            lambda:self.record['native_request'].update(added_count=True),
            lambda:self.record['native_request']['parameters'].update(logDefaultScale=50),
            lambda:self.record['native_argv'].append('--foreign')]
        for mutate in mutations:
            self.record=copy.deepcopy(original);mutate();self.persist()
            with self.assertRaises(ValueError):entry.model_source_gate(self.args)
        self.record=original;self.persist()
        target=self.case/'native/model-lattigo.mlir';target.unlink()
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)

    def test_generated_full_Q_P_scale_or_site_inventory_mismatch_rejects(self):
        for key,value in [('Q',[2]),('P',[3]),('log_default_scale',50)]:
            artifact=copy.deepcopy(self.artifact);artifact['parameters'][0][key]=value
            self.record['phases']['native_compile']['artifact']=artifact
            fixtures.put(self.case/'native/model-native-artifact.json',artifact);self.persist()
            with self.subTest(key=key),self.assertRaises(ValueError):entry.model_source_gate(self.args)
        self.record['phases']['native_compile']['artifact']=self.artifact
        fixtures.put(self.case/'native/model-native-artifact.json',self.artifact)
        self.record['static_native_bootstraps']=8;self.persist()
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)

    def test_closed_completion_registry_process_and_native_clock_required(self):
        with patch.object(entry.native,'_scan',return_value={501:dict(starttime=21,state='R')}):
            with self.assertRaises(ValueError):entry.model_source_gate(self.args)
        self.completion['registry_persisted']=False;self.persist()
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)
        self.completion['registry_persisted']=True;self.persist()
        with self.registry.open('a')as stream:stream.write(json.dumps(self.row)+'\n')
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)
        self.persist();self.record['phases']['native_compile']['observation']['ended_ns']=100;self.persist()
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)

    def test_gate_never_reads_native_IR_payload_or_invokes_native_builder(self):
        read=Path.read_text
        def metadata_only(path,*args,**kwargs):
            if path.suffix=='.mlir':raise AssertionError('MLIR payload read')
            return read(path,*args,**kwargs)
        with patch.object(Path,'read_text',metadata_only), \
                patch.object(entry.native,'native_command',side_effect=AssertionError('native builder')):
            entry.model_source_gate(self.args)

    def test_original_private_dispatch_three_tools_offline_and_codegen_only_after_build(self):
        namespace=entry._namespace(self.args)
        self.assertIs(namespace['execute'].__code__,entry.common.execute.__code__)
        self.assertIs(namespace['execute'].__globals__,namespace)
        self.assertIs(namespace['serial_lock'],entry.common.serial_lock)
        result=self.execute_entry()
        self.assertEqual(result['status'],'ok');self.assertTrue(result['go_library_compile_accepted'])
        self.assertTrue(result['code_generation_verified']);self.assertTrue(result['model_specific_native_codegen_verified'])
        self.assertTrue(result['inherited_native_target_IR_verified']);self.assertEqual(result['inherited_native_process_seconds'],1e-7)
        self.assertEqual(result['inherited_native_case_seconds'],1.5);self.assertFalse(result['previous_native_clock_added'])
        self.assertEqual(result['completed'],dict(utils_emission=1,main_emission=1,library_build=1))
        self.assertEqual(len(self.calls),3)
        for command,options in self.calls:
            self.assertEqual(options['mem_gib'],4);self.assertEqual(options['fsize_gib'],8)
            for key in ('GOPROXY','GOSUMDB','GOENV','GOWORK'):self.assertEqual(options['env'][key],'off')
            self.assertEqual(options['env']['GOTOOLCHAIN'],'local')
        self.assertEqual(result['timing_variant'],'fresh-Go-only-from-accepted-original-fusion-layout16384-native')
        self.assertFalse(result['generated_initializer_executed'])
        self.assertTrue(all(result[k]==0 for k in entry.common.ZERO_FIELDS))
        self.assertTrue(all(result[k]is None for k in entry.common.FORMAL_FIELDS))
        snapshots=entry.common.read(self.out/'source/snapshots.json')
        names={Path(x['source']).name for x in snapshots['files']}
        self.assertTrue({'heir_layout_16384_go_compile.py','heir_saved_math_layout_16384.py',
            'heir_no_fusion_go_compile.py','heir_saved_math_no_linear_fusion.py'}<=names)
        row=json.loads(self.registry.read_text().splitlines()[-1])
        self.assertEqual(set(row),entry.host_sources.REGISTRY_FIELDS)
        self.assertEqual(row['round'],entry.ROUND);self.assertTrue(row['metrics']['code_generation_verified'])

    def test_prepare_and_failed_Go_keep_accepted_native_fact_without_codegen(self):
        self.args.run=False;prepared=self.execute_entry()
        self.assertEqual(prepared['status'],'prepared');self.assertTrue(prepared['inherited_native_target_IR_verified'])
        self.assertFalse(prepared['code_generation_verified']);self.assertEqual(self.calls,[])
        self.args.run=True;self.args.out=self.root/'fresh-failed-Go'
        def failed(command,**options):
            fixtures.put(options['log'],'fake translator failure');return dict(fixtures.clean(),exit_code=1)
        failed=self.execute_entry(failed)
        self.assertEqual(failed['status'],'error');self.assertTrue(failed['inherited_native_target_IR_verified'])
        self.assertFalse(failed['code_generation_verified']);self.assertFalse(failed['library_build_passed'])
        self.assertEqual(json.loads((self.case/'result.json').read_text()),self.record)
        self.assertEqual((self.old_case/'result.json').read_text(),self.original_failed_record)

    def test_stuck_monitor_withdraws_Go_acceptance_retaining_build_and_native_facts(self):
        class Stuck:
            def __init__(self,**kwargs):pass
            def start(self):pass
            def is_alive(self):return True
            def join(self,timeout):pass
        with patch.object(entry.common.threading,'Thread',Stuck):result=self.execute_entry()
        self.assertEqual(result['status'],'error');self.assertFalse(result['monitor_stopped'])
        self.assertTrue(result['library_build_passed']);self.assertTrue(result['inherited_native_target_IR_verified'])
        self.assertFalse(result['go_library_compile_accepted']);self.assertFalse(result['code_generation_verified'])

    def test_original_backend_pin_and_parser_preserved(self):
        checked=entry.source_gate(self.args)
        self.assertEqual(checked['backend']['go_version'],'1.26.0');self.assertEqual(checked['backend']['lattigo_version'],'6.1.0')
        self.assertEqual(entry.common.TOTAL_SECONDS,1800)
        self.assertEqual(entry.common.PHASE_SECONDS,dict(utils_emission=300,main_emission=300,library_build=600))
        argv=['--model','alexnet','--source-variant','alex-serial-json','--source-case',str(self.case),
            '--out',str(self.out),'--backend-case',str(self.backend),'--cache-source',str(self.cache),
            '--translate',str(self.translate),'--go',str(self.go),'--run']
        self.assertEqual(entry.parser().parse_args(argv).source_variant,'alex-serial-json')
        fixtures.put(self.cache/'toolchain/go/VERSION','go1.26.1\n')
        with self.assertRaises(ValueError):entry.source_gate(self.args)


if __name__=='__main__':unittest.main()
