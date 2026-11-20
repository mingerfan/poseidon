"""Metadata-only guards for closed fresh-Agent and official compiler cases."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from pt_agent.experiments import completed_compile_archive as archive
from pt_agent.tests import test_completed_compile_archive as fixtures


class NewClosedCaseArchiveChecks(unittest.TestCase):
    def fixture(self, root, official=False):
        case, old_emit, _, _ = fixtures.CompletedCompileArchiveTests().fixture(root)
        clean = dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])
        emit = case / ('work/compile' if official else 'post-freeze/emit')
        emit.parent.mkdir(exist_ok=True)
        shutil.move(old_emit, emit)
        plan = emit / 'lowered._hecate_golden.runtime-plan.json'
        if official:
            trace = case / 'work/trace/traced/ResNet.mlir'
            trace.parent.mkdir(parents=True)
            shutil.move(emit/'candidate_trace.mlir', trace)
            result = dict(mode='original-source-native-profile-compile', status='ok', error=None,
                model='dacapo-original-resnet20-silu', original_source_variant=True,
                export_contract_compatible=True, plan=dict(path=str(plan)),
                trace_process=clean, compile_process=clean, compile_worker=dict(process=clean))
            names = ('status', 'trace_attempts', 'trace_completed', 'compile_attempts', 'compile_completed',
                     'native_compile_attempts', 'native_compile_completed', 'bootstrap_count',
                     'export_contract_compatible', 'native_execution_ready', 'predicted_latency_seconds',
                     'actual_ckks_latency_seconds', 'accuracy_acceptance', 'security_acceptance')
            result.update(dict.fromkeys(names[1:7],1))
        else:
            stages = ('intake','calibration','agent','placement','emit','trace','compile')
            phases = {n:dict(status='ok') for n in stages}
            phases['trace'].update(process=clean,mlir=dict(path=str(emit/'candidate_trace.mlir')))
            phases['compile'].update(process=clean,runtime_plan=str(plan))
            audit = case/'post-run-quality-audit.json'
            audit.write_text(json.dumps(dict(formal_frozen=True,quality_acceptance_passed=True,strict_math_passed=True)))
            result = dict(mode='fresh-common-model-agent-e2e',status='ok',error=None,
                model='resnet20_silu',attempted=dict.fromkeys(stages,1),completed=dict.fromkeys(stages,1),
                phases=phases,processes=[dict(observation=clean)for _ in stages],quality_audit=str(audit))
            names = ('status','attempted','completed','variant','api_delta','compiled_bootstraps',
                     'predicted_latency_us','CKKS_latency','accuracy_acceptance',
                     'recorded_elapsed_seconds','completion_path')
        (case/'result.json').write_text(json.dumps(result))
        row = dict(run_id=case.name,round='fixture',model=result['model'],mode=result['mode'],
                   metrics={n:result.get(n)for n in names},git_rev='fixture',artifacts_dir=str(case),ts='fixture')
        (root/'registry.jsonl').write_text(json.dumps(row)+'\n')
        return case, emit, result

    def test_fresh_case_keeps_plan_and_only_selects_current_derived_outputs(self):
        with TemporaryDirectory()as directory:
            root=Path(directory).resolve();case,emit,_=self.fixture(root)
            closed=archive.validate_completed_case(case/'result.json',results_root=root)
            self.assertEqual(closed['eligible_files']['emit_constants'],emit/'constants.npz')
            self.assertEqual(closed['eligible_bundle'].parent,emit)
            self.assertNotIn('quality_audit',closed['eligible_files'])
            with (root/'registry.jsonl').open('a')as stream:
                stream.write(json.dumps(dict(mode='llm-call',artifacts_dir=str(case)))+'\n')
            archive.validate_completed_case(case/'result.json',results_root=root)

    def test_fresh_unfrozen_or_unclean_phase_is_rejected(self):
        with TemporaryDirectory()as directory:
            root=Path(directory).resolve();case,_,result=self.fixture(root)
            audit=case/'post-run-quality-audit.json'
            audit.write_text(json.dumps(dict(formal_frozen=False,quality_acceptance_passed=True,strict_math_passed=True)))
            with self.assertRaises(ValueError):archive.validate_completed_case(case/'result.json',results_root=root)
            audit.write_text(json.dumps(dict(formal_frozen=True,quality_acceptance_passed=True,strict_math_passed=True)))
            result['phases']['compile']['process']['timed_out']=True
            (case/'result.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError):archive.validate_completed_case(case/'result.json',results_root=root)

    def test_official_case_selects_two_ir_files_and_retains_original_plan(self):
        with TemporaryDirectory()as directory:
            root=Path(directory).resolve();case,emit,result=self.fixture(root,official=True)
            closed=archive.validate_completed_case(case/'result.json',results_root=root)
            self.assertEqual(set(closed['eligible_files']),{'compiler_ir_input','compiler_ir_final'})
            self.assertEqual(closed['eligible_files']['compiler_ir_input'],case/'work/trace/traced/ResNet.mlir')
            self.assertEqual(closed['eligible_files']['compiler_ir_final'],emit/'lowered.mlir')
            result['export_contract_compatible']=False
            (case/'result.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError):archive.validate_completed_case(case/'result.json',results_root=root)

    def test_terminal_registry_metrics_must_match(self):
        with TemporaryDirectory()as directory:
            root=Path(directory).resolve();case,_,_=self.fixture(root,official=True)
            registry=root/'registry.jsonl';row=json.loads(registry.read_text());row['metrics']['status']='error'
            registry.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):archive.validate_completed_case(case/'result.json',results_root=root)

    def test_original_vgg_and_alex_keep_their_own_trace_identity(self):
        for model, entry in (('vgg16-silu-avgpool', 'VGG16'),
                             ('alexnet-silu-avgpool', 'AlexNet')):
            with self.subTest(model=model), TemporaryDirectory() as directory:
                root=Path(directory).resolve();case,_,result=self.fixture(root,official=True)
                before=case/'work/trace/traced/ResNet.mlir'
                trace=before.with_name(entry+'.mlir');before.rename(trace)
                result['model']='dacapo-original-'+model
                (case/'result.json').write_text(json.dumps(result))
                registry=root/'registry.jsonl';row=json.loads(registry.read_text())
                row['model']=result['model'];registry.write_text(json.dumps(row)+'\n')
                closed=archive.validate_completed_case(case/'result.json',results_root=root)
                self.assertEqual(closed['eligible_files']['compiler_ir_input'],trace)


if __name__=='__main__':unittest.main()
