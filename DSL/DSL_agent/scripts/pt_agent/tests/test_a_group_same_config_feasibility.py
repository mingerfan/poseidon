"""Metadata-only S0 projection, closed-source and failure publication fixtures."""
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import a_group_same_config_feasibility as d


def evidence():
    orbit=dict(qwen48_contract=dict(target=dict(d.PARAMETERS,Rule_slots=16384),gaps=[dict(fact='Parameter mapping gap')]),
        current_actions=dict(parser_or_solver_runs=0,new_model_Count_compile_download_patch_API_CKKS=0),
        models=[dict(model=m,classification='not_run_adapter_required',new_compiler_runs=0,same_Rule63_direct_ready=False)for m in d.MODELS],
        input_contract=dict(direct_gap='Dense constant representation gap'),cost_contract=dict(gaps=[dict(fact='Index and price gap')]),
        bootstrap_count_contract=dict(required='Decoded weighted execution count required'))
    heir=dict(requested_contract=dict(compiler_profile_fields=dict(d.PARAMETERS),logical_slots=16384,param_group='ours-N16-boot5'),
        activity=dict(new_native_compiles=0,new_model_exports=0,new_math_runs=0),
        rows=[dict(model=m,formal_run_status='not_run',scope_status='adapter_parameter_contract_gap',
            compiler_attempts_in_this_review=0,compiler_failure_recorded=False,expanded_execution_bootstraps=None,
            predicted_latency_s=None,adaptation_needed=['Independent initial8 and bootout5 management gap'],
            original_Rule63_arithmetic_adapter_reusable=True,missing_price_ops=['relinearize'])for m in d.MODELS],
        source_interfaces=dict(profile_prediction_gap='Typed execution cost mapping gap'),semantic_gap=dict(required='Original1to5 contract'))
    return orbit,heir


class FeasibilityMetadata(unittest.TestCase):
    def setup_sources(self,root):
        cases=[];registry=[]
        for model in d.MODELS:
            case=root/('synthetic-a2b-'+model);case.mkdir();cases.append(case)
            result=dict(round='COLD-RULE63',run_id=case.name,model=d.FACTORIES[model],placement='depth',
                status='error',error='Synthetic clean source failure',monitor_stopped=True,completed={})
            (case/'result.json').write_text(json.dumps(result))
            (case/'completion.json').write_text(json.dumps(dict(status='error',registry_persisted=True,required_artifacts_persisted=True)))
            registry.append(dict(run_id=case.name,round='COLD-RULE63',model=d.FACTORIES[model],mode='fixture',git_rev='synthetic',
                metrics=dict(status='error'),artifacts_dir=str(case),ts='synthetic'))
        (root/'registry.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in registry))
        orbit,heir=evidence();op=root/'orbit.json';hp=root/'heir.json'
        op.write_text(json.dumps(orbit));hp.write_text(json.dumps(heir))
        return SimpleNamespace(out=root/'s0case',destination=root/'publication',source_case=cases,orbit_evidence=op,heir_evidence=hp)

    def test_projection_preserves_six_unrun_null_results_and_exact_parameters(self):
        orbit,heir=evidence();note=d.project(orbit,heir,orbit_path='orbit.json',heir_path='heir.json')
        self.assertEqual(len(note['rows']),6);self.assertEqual(note['compiler_attempts'],0)
        for row in note['rows']:
            self.assertEqual(row['formal_run_status'],'not_run');self.assertEqual(row['diagnostic_status'],'adapter_gap')
            self.assertIsNone(row['compile_s']);self.assertIsNone(row['expanded_execution_bootstraps'])
            self.assertIsNone(row['pred_latency_s']);self.assertFalse(row['compiler_failure_recorded'])
            self.assertFalse(row['official_system_unsupported_proven'])
            if row['system']=='heir':
                self.assertEqual(row['source_reported_missing_price_ops'],['relinearize'])
                self.assertEqual(row['missing_price_ops'],[])
                self.assertEqual(row['policy_excluded_ops'],['relinearize'])
                self.assertTrue(row['policy_excluded_relin'])
        heir['requested_contract']['compiler_profile_fields']['bootstrapLevelUpperBound']=16
        with self.assertRaises(ValueError):d.project(orbit,heir,orbit_path='orbit.json',heir_path='heir.json')

    def test_clean_failed_sources_allow_note_and_unique_metadata_case_without_paper_rows(self):
        from pt_agent import paths,registry
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);args=self.setup_sources(root)
            original=d.persist_case_completion
            def persistence(*a,**k):
                signal.raise_signal(signal.SIGTERM);signal.raise_signal(signal.SIGALRM)
                return original(*a,**k)
            with patch.object(paths,'RESULTS_DIR',root),patch.object(registry,'git_rev',return_value='synthetic'),\
                    patch.object(d,'persist_case_completion',side_effect=persistence):
                result=d.run(args)
            self.assertEqual(result['status'],'ok');self.assertEqual(result['compiler_attempts'],0)
            self.assertTrue(all(r['status']=='error'for r in result['source_cases']))
            note=json.loads((args.destination/'s0_same_config_feasibility.json').read_text())
            self.assertEqual(len(note['rows']),6);self.assertFalse(note['formal_S1_results_available'])
            self.assertTrue((args.destination/'s0_same_config_feasibility.html').is_file())
            self.assertFalse((args.destination/'runs.jsonl').exists())
            rows=[json.loads(l)for l in(root/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),4);self.assertEqual(len(rows[-1]),8)
            self.assertEqual(rows[-1]['run_id'],args.out.name)
            self.assertEqual(json.loads((args.out/'completion.json').read_text())['status'],'ok')

    def test_unclosed_or_duplicate_source_registry_stops_note_but_saves_own_error(self):
        from pt_agent import paths,registry
        for failure in('source_running','duplicate_registry'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory()as tmp:
                root=Path(tmp);args=self.setup_sources(root)
                if failure=='source_running':
                    p=args.source_case[0]/'result.json';x=json.loads(p.read_text());x['status']='running';p.write_text(json.dumps(x))
                else:
                    p=root/'registry.jsonl';p.write_text(p.read_text()+p.read_text().splitlines()[0]+'\n')
                with patch.object(paths,'RESULTS_DIR',root),patch.object(registry,'git_rev',return_value='synthetic'):
                    result=d.run(args)
                self.assertEqual(result['status'],'error');self.assertEqual(result['diagnostic_failure_category'],'driver_error')
                self.assertEqual(result['compiler_attempts'],0)
                self.assertFalse((args.destination/'s0_same_config_feasibility.json').exists())
                self.assertEqual(json.loads((args.out/'completion.json').read_text())['status'],'error')


if __name__=='__main__':unittest.main(verbosity=2)
