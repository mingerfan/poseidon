"""Closed continued compilation provenance; synthetic paths, no real payloads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import completed_compile_archive as m


class ContinuedAlexArchive(unittest.TestCase):
    def fixture(self,root):
        case=root/'compiled';emit=case/'post-freeze/emit';emit.mkdir(parents=True)
        plan=emit/'lowered.runtime-plan.json';plan.write_text('{}')
        trace=emit/'candidate_trace.mlir';trace.write_text('synthetic')
        result=dict(mode='accepted-alex-continuation-trusted-postfreeze-compile',
            status='ok',error=None,model='alexnet_silu_avgpool',compiled_bootstraps=12,
            attempted={'compile':1},completed={'compile':1},source_case=str(root/'frozen'),
            phases={'compile':{'runtime_plan':str(plan)},'trace':{'mlir':{'path':str(trace)}}})
        complete=dict(status='ok',required_artifacts_persisted=True,registry_persisted=True,
            timing_status='ok',overlap_and_timing_diagnostics=[])
        names=('status','attempted','completed','compiled_bootstraps','predicted_latency_us',
               'incoming_signal','guard_trigger','source_case')
        row=dict(run_id=case.name,round='test',model=result['model'],mode=result['mode'],git_rev=None,
            metrics={k:result.get(k) for k in names},artifacts_dir=str(case),ts='test')
        (root/'registry.jsonl').write_text(json.dumps(row)+'\n')
        for name,data in [('result.json',result),('completion.json',complete)]:
            (case/name).write_text(json.dumps(data))
        gate=dict(compile_case=str(case),clean_native_compile=True,compiled_input_logical_path=str(trace))
        return case,result,complete,row,gate

    def test_derived_outputs_only_and_dispatch_reuses_complete_source_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,r,c,row,gate=self.fixture(root)
            with patch('pt_agent.experiments.r30_alex_postfreeze.compiled_source',return_value=gate) as checked:
                closed=m.validate_completed_case(case/'result.json',results_root=root)
            checked.assert_called_once_with(case)
            self.assertEqual(set(closed['eligible_files']),{'compiler_ir_input','compiler_ir_final'})
            self.assertEqual(closed['eligible_bundle'],case/'post-freeze/emit/lowered.bundle')

    def test_source_gate_rejection_preserves_all_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,r,c,row,gate=self.fixture(root)
            with patch('pt_agent.experiments.r30_alex_postfreeze.compiled_source',side_effect=ValueError('unfrozen')):
                with self.assertRaises(ValueError):m._validate_alex_postfreeze(case,r,c,root)
            self.assertTrue((case/'post-freeze/emit/candidate_trace.mlir').is_file())

    def test_nonterminal_or_unpersisted_completion_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,r,c,row,gate=self.fixture(root)
            for change in ({'status':'error'},{'registry_persisted':False},{'timing_status':'error'}):
                with self.assertRaises(ValueError):m._validate_alex_postfreeze(case,r,dict(c,**change),root)

    def test_changed_registry_counts_and_duplicate_record_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,r,c,row,gate=self.fixture(root)
            with patch('pt_agent.experiments.r30_alex_postfreeze.compiled_source',return_value=gate):
                row['metrics']['compiled_bootstraps']=13
                (root/'registry.jsonl').write_text(json.dumps(row)+'\n')
                with self.assertRaises(ValueError):m._validate_alex_postfreeze(case,r,c,root)
                row['metrics']['compiled_bootstraps']=12
                (root/'registry.jsonl').write_text((json.dumps(row)+'\n')*2)
                with self.assertRaises(ValueError):m._validate_alex_postfreeze(case,r,c,root)

    def test_foreign_trace_or_plan_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,r,c,row,gate=self.fixture(root)
            with patch('pt_agent.experiments.r30_alex_postfreeze.compiled_source',return_value=gate):
                r['phases']['trace']['mlir']['path']=str(root/'weights.npz')
                with self.assertRaises(ValueError):m._validate_alex_postfreeze(case,r,c,root)


if __name__=='__main__':unittest.main()
