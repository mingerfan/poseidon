import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import a5_a6_rule63_search as own
from pt_agent.experiments import r29_local_compile_matrix as original
from pt_agent.experiments.r30_common_model_e2e import DIAGNOSTIC_ELIDE_OPTION
from pt_agent.tests._artifacts import require
from pt_agent.experiments.r10_paper_metrics import summarize_events


class RuleSearchTests(unittest.TestCase):
    def test_parent_closure_persists_one_registry_and_child_clock_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'parent';child=out/'dacapo-optimized';child.mkdir(parents=True)
            (child/'completion.json').write_text(json.dumps(dict(status='ok',registry_persisted=True,
                required_artifacts_persisted=True,complete_case_elapsed_seconds=3)))
            args=SimpleNamespace(out=out,model='vgg16',git_rev=None)
            row=dict(status='ok',status_detail='finite fixture',compile_s=2)
            registry=Path(temp)/'registry.jsonl';timing=summarize_events([])
            result=own.close_parent(args,child,dict(status='ok',compiled_bootstraps=7),row,
                started_ns=time.perf_counter_ns()-1000,timing=timing,registry=registry)
            self.assertEqual(result['status'],'ok');self.assertEqual(result['compile_s'],2)
            self.assertIsNone(result['generation_e2e_s'])
            self.assertEqual(result['child_completion'],str(child/'completion.json'))
            entry=json.loads(registry.read_text());self.assertEqual(len(entry),8)
            self.assertTrue(json.loads((out/'completion.json').read_text())['registry_persisted'])
            with self.assertRaises(ValueError):
                own.close_parent(args,child,dict(status='ok'),row,
                    started_ns=time.perf_counter_ns(),timing=timing,registry=registry)

    def test_private_native_context_enters_and_restores_original_toolchain(self):
        from pt_agent import toolchain
        before = toolchain.compile_argv
        ns = own.namespace(SimpleNamespace(diagnostic_elide_final_ir=True))
        template = ['{hecate_opt}', '{input}', '--pars', '--waterline=43',
            '--runtime-plan-context-id=qwen-bootstrap48',
            '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
            '--runtime-plan-operator-spec-sha256=sha256:'+'a'*64,
            '-o', '{output_dir}/lowered.mlir']
        with ns['native_metadata_commands'](template):
            self.assertIsNot(toolchain.compile_argv, before)
            command = toolchain.compile_argv('dacapo-optimized', 'qwen48', '/fixture')
            self.assertEqual(command[0], '/hecate-opt')
            self.assertIn(DIAGNOSTIC_ELIDE_OPTION, command)
        self.assertIs(toolchain.compile_argv, before)

    def test_inner_worker_route_admits24_without_changing_old_launcher(self):
        args = SimpleNamespace(compile_seconds=7200, trace_seconds=1800, memory_gib=24,
            case_gib=64, model_info=Path('/model.json'), metadata_template=Path('/template.json'),
            profile=Path('/profile.json'), frontend=Path('/frontend.py'),
            model_factory=own.FACTORIES['vgg16'], diagnostic_elide_final_ir=True)
        before, wall = original._launch_worker(args,'dacapo-optimized','compile',Path('/program'),Path('/work'))
        after, new_wall = own.worker_command(args,'dacapo-optimized','compile',Path('/program'),Path('/work'))
        expected = [own.MODULE if x=='pt_agent.experiments.r29_local_compile_matrix' else x for x in before]
        self.assertEqual(after,expected+['--diagnostic-elide-final-ir'])
        self.assertEqual(wall,new_wall)
        self.assertEqual(after[after.index('--memory-gib')+1],'24')

    def test_compiler_argv_preserves_algorithm_and_only_elides_printing(self):
        template = ['{hecate_opt}','{input}','--pars','--runtime-plan-format=json',
            '--runtime-plan-bundle-format=files','--ckks-config={package}/config/compiler-config.json',
            '--waterline=43','--runtime-plan-context-id=qwen-bootstrap48',
            '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
            '--runtime-plan-operator-spec-sha256=sha256:'+'a'*64,
            '-o','{output_dir}/lowered.mlir']
        kwargs=dict(binary='/hecate',source='/trace',profile='/qwen48.json',output='/lowered.mlir')
        expected=original.compiler_argv(template,'dacapo-optimized',**kwargs)
        observed=own.compiler_command(template,'dacapo-optimized',diagnostic_elide_final_ir=True,**kwargs)
        self.assertEqual([a for a in observed if a!=DIAGNOSTIC_ELIDE_OPTION],expected)
        self.assertIn('--boot-placement=optimized',observed)
        self.assertIn('--waterline=43',observed)

    def test_private_namespace_does_not_modify_original_functions(self):
        before=(original.compiler_argv,original._launch_worker,original.run_process)
        args=SimpleNamespace(diagnostic_elide_final_ir=False)
        ns=own.namespace(args,outer=True)
        self.assertIs(ns['run_case'].__code__,original.run_case.__code__)
        self.assertIs(ns['worker'].__code__,original.worker.__code__)
        self.assertIs(ns['run_process'],own.guarded_process)
        self.assertEqual(before,(original.compiler_argv,original._launch_worker,original.run_process))

    def test_physical_guard_separates_rss_and_host_reserve(self):
        with patch.object(own,'latest_resource',return_value=dict(subtree_plus_supervisor_rss_bytes=15*own.GIB)), \
             patch.object(own,'available_ram',return_value=2*own.GIB):
            self.assertEqual(own.physical_guard_observation('/unused')['reasons'],
                             ['subtree including supervisors exceeds14GiB RSS'])
        with patch.object(own,'latest_resource',return_value=None), \
             patch.object(own,'available_ram',return_value=own.GIB):
            self.assertTrue(own.physical_guard_observation('/unused')['stop'])

    def test_terminal_classification_retains_crash_timeout_and_driver_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(own.terminal_category(dict(status='ok'),temp)[0],'ok')
            case=dict(status='error',phases={'compile':{'process':{'exit_code':-11,'signal':11}}})
            self.assertEqual(own.terminal_category(case,temp)[0],'compiler_crash')
            case['phases']['compile']['process']={'timed_out':True}
            self.assertEqual(own.terminal_category(case,temp)[0],'timeout')
            case['phases']['compile']['process']={'exit_code':0}
            self.assertEqual(own.terminal_category(case,temp)[0],'driver_error')
            path=Path(temp)/'compile-resources.physical-guard.json';path.write_text(json.dumps(dict(stop=True,reasons=['RSS'])))
            self.assertEqual(own.terminal_category(case,temp)[0],'resource_limit')
            path.write_text(json.dumps(dict(stop=True,error='own monitor error')))
            self.assertEqual(own.terminal_category(case,temp)[0],'driver_error')

    def test_sampled_subtree_peak_uses_concurrent_sums(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'compile-resources.jsonl'
            path.write_text(json.dumps(dict(processes=[dict(vmrss_kib=100),dict(vmrss_kib=200)],
                                            outer_process=dict(vmrss_kib=50)))+'\n')
            self.assertEqual(own.sampled_subtree_peak(temp),350/1024**2)
            self.assertIsNone(own.sampled_subtree_peak(Path(temp)/'absent'))

    def test_paper_row_uses_only_exclusive_native_span(self):
        with tempfile.TemporaryDirectory() as temp:
            args=SimpleNamespace(out=Path(temp)/'new-vgg',model='vgg16',diagnostic_elide_final_ir=True)
            record=dict(status='ok',compiled_bootstraps=17,phases={'compile':dict(execution_steps=81,
                timing={'spans':[dict(stage_id='native_compiler_7200s',exclusive_ns=4000000000),
                                 dict(stage_id='unrelated_parent',exclusive_ns=999000000000)]})})
            row=own.paper_record(args,record,temp)
            self.assertEqual(row['compile_s'],4)
            self.assertEqual(row['exec_ops'],81)
            self.assertIsNone(row['generation_e2e_s'])
            self.assertIsNone(row['pred_latency_s'])
            self.assertIsNone(row['peak_rss_gib'])
            self.assertEqual(row['api_calls'],0)


_SOURCES=[own.paths.RESULTS_DIR/'r17a-margin-heldout-20261115'/factory.split(':')[1]/'result.json'
          for factory in own.FACTORIES.values()]

@require(_SOURCES)
class RetainedSourceTests(unittest.TestCase):
    def test_two_original_retained_rule63_sources_metadata_only(self):
        for model,factory in own.FACTORIES.items():
            name=dict(vgg16='vgg16-silu-avgpool',alexnet='alexnet-silu-avgpool')[model]
            args=SimpleNamespace(model_factory=factory,
                rule_result=own.paths.RESULTS_DIR/'r17a-margin-heldout-20261115'/factory.split(':')[1]/'result.json',
                model_info=own.paths.RESULTS_DIR/('r13-common-'+name+'-20261115/worker/model_info/model_info.json'))
            checked=own.fixed63_source(args)
            self.assertEqual(checked['degree'],63)
            self.assertEqual(checked['activation_count'],15 if model=='vgg16' else 7)
            self.assertEqual(checked['new_quality_evaluations'],0)
