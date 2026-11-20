import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import a_group_resnet_artifacts as own


def process():
    return dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[],signals_sent=[])


class ResNetArtifactAdmissionTests(unittest.TestCase):
    def test_three_source_record_schemas_keep_original_parameter_groups_and_clean_compiler(self):
        for source in own.SOURCES:
            with self.subTest(source=source['name']):
                path=own.paths.RESULTS_DIR/source['case']/source['plan']
                if source['name']=='dacapo_native':
                    record=dict(status='ok',model=source['model'],plan=dict(path=str(path)),
                        execution_instructions=source['ops'],bootstrap_count=source['boots'],
                        source_checked=True,original_plan_modified=False,compile_process=process(),
                        compile_worker=dict(status='ok',process=process()))
                else:
                    record=dict(status='ok',model=source['model'],compiled_bootstraps=source['boots'],
                        phases=dict(compile=dict(status='ok',runtime_plan=str(path),execution_steps=source['ops'],
                            compiled_bootstraps=source['boots'],process=process())))
                self.assertEqual(own.validate_record(source,record,source['parameters']),path)
                changed=dict(source['parameters'],rescalingFactor=source['parameters']['rescalingFactor']+1)
                with self.assertRaises(ValueError):own.validate_record(source,record,changed)
                record['model']='foreign model'
                with self.assertRaises(ValueError):own.validate_record(source,record,source['parameters'])

    def test_closed_fixture_counts_and_host_birth_checks_are_required_before_export(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);case=root/'fixture';case.mkdir();plan=case/'plan.json'
            source=dict(name='fixed',case='fixture',run_id='fixture-run',plan='plan.json',profile='profile.json',
                model='fixture-model',system='fixed_rule_depth',param_group='ours-N16-boot5',boots=1,ops=2,parameters=own.OURS)
            values=[dict(id='x',level=1),dict(id='y',level=5),dict(id='z',level=5)]
            execution=[dict(op='boot',ordinal=1,inputs=['x'],output='y'),dict(op='negate',ordinal=2,inputs=['y'],output='z')]
            plan.write_text(json.dumps(dict(values=values,execution=execution)))
            record=dict(status='ok',model='fixture-model',compiled_bootstraps=1,source=dict(scope='public fixture'),
                phases=dict(compile=dict(status='ok',runtime_plan=str(plan),compiled_bootstraps=1,
                    execution_steps=2,process=process())))
            (case/'result.json').write_text(json.dumps(record))
            (case/'completion.json').write_text(json.dumps(dict(status='ok',registry_persisted=True,required_artifacts_persisted=True)))
            (case/'profile.json').write_text(json.dumps(own.OURS))
            (case/'test-resources.jsonl').write_text(json.dumps(dict(processes=[dict(pid=90001,starttime=42)]))+'\n')
            row=dict(run_id='fixture-run',round='fixture',model='fixture-model',mode='fixture',git_rev=None,
                metrics=dict(status='ok'),artifacts_dir=str(case),ts='fixture')
            (root/'registry.jsonl').write_text(json.dumps(row)+'\n')
            with patch.object(own.paths,'RESULTS_DIR',root),patch.object(own,'same_birth',return_value=False):
                admitted=own.admit_source(source,root/'destination',check_births=True)
            self.assertTrue(admitted['host_births_checked']);self.assertEqual(admitted['observed']['boots'],1)
            self.assertEqual(admitted['observed']['bootstrap_level_pairs'],[dict(input_levels=[1],output_level=5,count=1)])
            self.assertFalse((root/'destination').exists())
            with patch.object(own.paths,'RESULTS_DIR',root),patch.object(own,'same_birth',return_value=True):
                with self.assertRaises(ValueError):own.admit_source(source,root/'destination',check_births=True)
            execution.pop();plan.write_text(json.dumps(dict(values=values,execution=execution)))
            with patch.object(own.paths,'RESULTS_DIR',root):
                with self.assertRaises(ValueError):own.admit_source(source,root/'destination',check_births=False)


if __name__=='__main__':unittest.main()
