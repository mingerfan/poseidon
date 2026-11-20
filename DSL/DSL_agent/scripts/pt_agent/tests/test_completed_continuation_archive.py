import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import completed_compile_archive as archive


class ClosedContinuationArchive(unittest.TestCase):
    def fixture(self,root):
        case=root/'continued';emit=case/'emit';emit.mkdir(parents=True)
        plan=emit/'lowered.native.runtime-plan.json';plan.write_text('{}')
        observation=dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])
        source=dict(original_case=str(root/'retained'),trace_metadata=dict(path=str(root/'retained/work/trace/traced/VGG16.mlir')))
        result=dict(status='ok',error=None,mode='saved-original-trace-single-compile',model='dacapo-original-vgg16-silu-avgpool',
            original_trace_reused=True,trace_attempts=0,compile_attempts=1,compile_completed=1,native_compile_attempts=1,
            native_compile_completed=1,export_contract_compatible=True,guard_trigger=None,bootstrap_count=20,
            processes=[dict(observation=observation)],compile_worker=dict(process=observation),source=source,plan=dict(path=str(plan)))
        complete=dict(status='ok',registry_persisted=True,required_artifacts_persisted=True,timing_status='ok',overlap_and_timing_diagnostics=[])
        names=('status','compile_attempts','compile_completed','native_compile_attempts','native_compile_completed',
            'bootstrap_count','export_contract_compatible','actual_ckks_latency_seconds','security_acceptance')
        row=dict(run_id=case.name,round='test',model=result['model'],mode=result['mode'],git_rev=None,
            metrics={k:result.get(k)for k in names},artifacts_dir=str(case),ts='test')
        (root/'registry.jsonl').write_text(json.dumps(row)+'\n')
        return case,result,complete,source,row

    def test_source_and_compiler_outputs_have_explicit_separate_origins(self):
        with TemporaryDirectory()as tmp:
            root=Path(tmp);case,result,complete,source,row=self.fixture(root)
            with patch('pt_agent.experiments.dacapo_compile_continuation.source_gate',return_value=source):
                closed=archive._validate_compile_continuation(case,result,complete,root)
            self.assertEqual(closed['eligible_files']['compiler_ir_input'],Path(source['trace_metadata']['path']))
            self.assertEqual(closed['eligible_files']['compiler_ir_final'],case/'emit/lowered.mlir')
            self.assertNotIn('model',closed['eligible_files'])

    def test_failed_compile_or_guard_is_ineligible(self):
        for changes in [dict(compile_completed=0),dict(guard_trigger={'kind':'rss'})]:
            with self.subTest(changes=changes),TemporaryDirectory()as tmp:
                root=Path(tmp);case,result,complete,source,row=self.fixture(root);result.update(changes)
                with self.assertRaises(ValueError):archive._validate_compile_continuation(case,result,complete,root)

    def test_reconstructed_provenance_must_match(self):
        with TemporaryDirectory()as tmp:
            root=Path(tmp);case,result,complete,source,row=self.fixture(root);changed=copy.deepcopy(source);changed['original_case']='changed'
            with patch('pt_agent.experiments.dacapo_compile_continuation.source_gate',return_value=changed),self.assertRaises(ValueError):
                archive._validate_compile_continuation(case,result,complete,root)

    def test_unique_registry_metrics_required(self):
        with TemporaryDirectory()as tmp:
            root=Path(tmp);case,result,complete,source,row=self.fixture(root);row['metrics']['bootstrap_count']=21
            (root/'registry.jsonl').write_text(json.dumps(row)+'\n')
            with patch('pt_agent.experiments.dacapo_compile_continuation.source_gate',return_value=source),self.assertRaises(ValueError):
                archive._validate_compile_continuation(case,result,complete,root)


if __name__=='__main__':unittest.main()
