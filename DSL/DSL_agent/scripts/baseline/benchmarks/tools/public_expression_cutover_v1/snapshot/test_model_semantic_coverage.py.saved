"""Offline model capability and approved advanced-cohort evidence checks."""
import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from custom_batch_manifest import load_manifest
from model_semantic_coverage import analyze_graph, summarize_models, OP_SEMANTICS
from model_graph import OPS

BASE = Path(__file__).parent
MANIFEST = BASE/'cases/advanced-shapes-manifest.json'


class ModelSemanticTests(unittest.TestCase):
    def test_explicit_ten_models_and_no_catalog_name_dispatch(self):
        rows, _, _ = load_manifest(MANIFEST)
        self.assertEqual(len(rows), 10)
        for row in rows:
            model = row['descriptor']
            actual = analyze_graph(model)
            renamed = copy.deepcopy(model)
            renamed['id'] = 'user-chosen-name'
            self.assertEqual(analyze_graph(renamed), actual)
        observed = [analyze_graph(r['descriptor']) for r in rows]
        self.assertEqual({s['linear_shapes'][0][1] for s in observed[:4]}, {5,6,7,8})
        self.assertTrue(all(s['linear_depth'] == 2 for s in observed[:4]))
        features = {f for s in observed for f in s['features']}
        self.assertTrue({'conv1d.groups.2','conv1d.dilation.3','conv2d.dilation.1x2',
                         'input.shape.2x1x2','broadcast.exact.8'} <= features)

    def test_dead_nodes_do_not_inflate_semantics_and_invalid_graphs_fail(self):
        model = dict(schema=2,id='not-a-catalog',input_shape=[4],constants={'c':.5},
            nodes=[dict(id='unused',op='rotate',inputs=['x'],step=-3),
                   dict(id='used',op='add',inputs=['x','c'])],output='used')
        result = analyze_graph(model)
        self.assertEqual(result['operators'], {'add':1})
        self.assertEqual(result['dead_nodes'], ['unused'])
        self.assertNotIn('rotation.step.-3',result['features'])
        model['nodes'][0]['step']=99
        with self.assertRaises(ValueError): analyze_graph(model)
        with self.assertRaises(ValueError): analyze_graph(dict(schema=1,id='linear-0',family='linear',configuration=0))

    def test_model_power_and_scalar_broadcast_are_not_inferred_from_dsl_spelling(self):
        model=dict(schema=2,id='explicit-power',input_shape=[4],constants={'one':[.5]},
            nodes=[dict(id='p',op='power',inputs=['x'],exponent=4),
                   dict(id='out',op='subtract',inputs=['p','one'])],output='out')
        result=analyze_graph(model)
        self.assertEqual(result['operators'], {'power':1,'subtract':1})
        self.assertTrue({'power.exponent.4','broadcast.length1'} <= set(result['features']))
        self.assertEqual(set(OP_SEMANTICS), set(OPS))
        summary=summarize_models([dict(evidence='/saved/a',model_semantics=result)])
        self.assertEqual(len(summary['operator_matrix']),16)
        self.assertFalse(summary['full_model_domain_verified'])

    def test_legacy_catalog_is_retained_but_not_claimed_free_graph(self):
        from audit_model_capabilities import audit_models
        legacy=dict(schema=1,id='linear-0',family='linear',configuration=0)
        checked=dict(cases=[dict(evidence='/results/a',case='linear-0')],cohorts=[],model_cases=1,
                     numerical_coverage_status='all_cohorts_complete')
        with patch('audit_model_capabilities.audit_cohorts',return_value=checked), patch(
                'audit_model_capabilities.metadata',return_value=(legacy,'a'*64)):
            result=audit_models(['/results/batch'])
        self.assertEqual((result['total_agent_cases'],result['graph_cases'],result['legacy_catalog_cases']),(1,0,1))
        self.assertEqual(set(result['missing_graph_operators']),set(OPS))


@unittest.skipUnless(os.environ.get('POSEIDON_ADVANCED_AGENT_REPORT'), 'requires actual advanced live cohort')
class AdvancedAgentEvidenceTests(unittest.TestCase):
    def test_ten_real_passes_same_user_weights_and_parameter_coverage(self):
        from audit_agent_lineage import audit, metadata
        from audit_model_capabilities import audit_models
        from hecate_python_env import WORK
        path=Path(os.environ['POSEIDON_ADVANCED_AGENT_REPORT'])
        report=audit(path)
        rows,_,_=load_manifest(MANIFEST)
        models={r['descriptor']['id']:r['descriptor'] for r in rows}
        self.assertEqual((report['planned'],report['passed']), (10,10))
        self.assertEqual(report['input_executions'],40)
        self.assertEqual(report['private_key_directories_retained'],0)
        self.assertEqual(set(r['case'] for r in report['cases']),set(models))
        for row in report['cases']:
            self.assertEqual((row['provider'],row['model']),('deepseek','deepseek-flash'))
            run=Path(row['evidence'])
            model,_=metadata(run/'model.json',WORK/'results')
            self.assertEqual(model,models[row['case']])
        coverage=audit_models([path])
        self.assertEqual((coverage['graph_cases'],coverage['legacy_catalog_cases']),(10,0))
        features={r['feature'] for r in coverage['feature_matrix']}
        self.assertTrue({f'linear.output_width.{w}' for w in (5,6,7,8)} <= features)
        self.assertTrue({'conv1d.groups.2','conv1d.dilation.3','conv2d.dilation.1x2'} <= features)
        self.assertFalse(coverage['all_goal_requirements_complete'])


@unittest.skipUnless(os.environ.get('POSEIDON_MODEL_CAPABILITY_REPORT'),'requires saved 114-case capability audit')
class ModelCapabilityEvidenceTests(unittest.TestCase):
    def test_saved_free_graph_and_legacy_counts_are_separate(self):
        from audit_agent_lineage import metadata,read
        from audit_model_capabilities import audit_models
        from hecate_python_env import WORK
        report,_=metadata(Path(os.environ['POSEIDON_MODEL_CAPABILITY_REPORT']),WORK/'results')
        self.assertEqual((report['total_agent_cases'],report['graph_cases'],report['legacy_catalog_cases']),(114,66,48))
        self.assertEqual(report['missing_graph_operators'],['power'])
        self.assertFalse(report['full_model_domain_verified'])
        self.assertFalse(report['tests_executed_by_summary'])
        from historical_model_analysis import verify_sources,compare_historical
        verify_sources(report,BASE)
        current=audit_models([Path(c['report']) for c in report['cohorts']])
        compare_historical(report,current)
        self.assertEqual(set(current['missing_graph_operators']),{'batch_norm','power','reshape','concat'})
        self.assertEqual(set(r['operator'] for r in report['operator_matrix']),set(OPS)-{'batch_norm','reshape','concat'})


if __name__ == '__main__':
    unittest.main()
