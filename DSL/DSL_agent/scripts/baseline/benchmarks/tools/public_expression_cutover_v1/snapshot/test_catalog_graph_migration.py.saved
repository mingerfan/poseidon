"""New self-contained fixtures do not inherit old Agent success claims."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BASE=Path(__file__).parent


class RuleManifestCliTests(unittest.TestCase):
    def test_explicit_suite_plan_never_selects_catalog_or_reads_credentials(self):
        import contextlib
        import io
        from unittest.mock import patch
        import run_agent_batch
        from custom_batch_manifest import load_manifest
        for filename,count in [('legacy48-user-graphs.json',48),('user-graph-suite-v2.json',96)]:
            path=BASE/'cases'/filename
            rows,manifest,_=load_manifest(path)
            output=io.StringIO()
            with patch('sys.argv',['batch','--plan','--case-manifest',str(path)]), \
                 patch('run_agent_batch.batch_plan',side_effect=AssertionError('catalog lookup')), \
                 patch('agent_credentials.load_api_key',side_effect=AssertionError('credential access')), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(run_agent_batch.main(),0)
            plan=json.loads(output.getvalue())
            self.assertEqual(plan['cases'],rows)
            self.assertEqual(len(manifest['cases']),count)
            self.assertEqual(plan['agent_calls'],0)
            self.assertEqual((plan['service_provider'],plan['model'],plan['api_concurrency']),
                             ('deepseek','deepseek-flash',10))

    def test_ambiguous_or_silently_reduced_manifests_are_rejected(self):
        manifest=BASE/'cases/advanced-shapes-manifest.json'
        for flags in (['--case-manifest',str(manifest),'--smoke'],
                      ['--case-manifest',str(manifest),'--extended'],
                      ['--case-manifest',str(manifest),'--unit-tests'],
                      ['--case-manifest',str(manifest),'--worker','unused'],
                      ['--case-manifest',str(manifest),'--keys','unused'],
                      ['--manifest-sha256','a'*64]):
            result=subprocess.run([sys.executable,'-B',str(BASE/'run_model_batch.py'),*flags],
                                  capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,2)

    def test_changed_missing_or_unreadable_manifest_fails_closed(self):
        from unittest.mock import patch
        from run_model_batch import frozen_manifest_valid
        from hecate_python_env import digest
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json'
            path.write_text('{}')
            expected=digest(path)
            self.assertTrue(frozen_manifest_valid(path,expected))
            path.write_text('{"changed":true}')
            self.assertFalse(frozen_manifest_valid(path,expected))
            with patch('run_model_batch.digest',side_effect=OSError('unreadable')):
                self.assertFalse(frozen_manifest_valid(path,expected))
            path.unlink()
            self.assertFalse(frozen_manifest_valid(path,expected))


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires pinned Torch')
class CatalogMigrationTorchTests(unittest.TestCase):
    def test_48_exports_match_original_torch_reference_and_rule_lowering(self):
        import numpy as np
        import torch
        from catalog_graph_migration import expand_legacy
        from model_catalog import descriptors,build_model,test_inputs
        from model_graph import build_graph_model,evaluate_reference
        from fx_to_hecate import translate
        from test_fx_to_hecate import interpret_fragment
        from unittest.mock import patch
        for old in descriptors():
            before=copy.deepcopy(old)
            graph=expand_legacy(old)
            self.assertEqual(old,before)
            self.assertEqual(set(graph),{'schema','id','input_shape','constants','nodes','output'})
            self.assertEqual(graph['schema'],2)
            original,shape=build_model(old)
            # After export neither execution nor lowering may consult the catalog.
            with patch('model_catalog.CatalogModel',side_effect=AssertionError('catalog lookup after export')):
                native,new_shape=build_model(graph)
                renamed=copy.deepcopy(graph)
                renamed['id']='user-provided-name'
                user_model,_=build_graph_model(renamed)
                payload=translate(native,new_shape)
                inputs=list(test_inputs(shape))
                rng=np.random.default_rng(1701)
                inputs.extend(rng.uniform(-1,1,size=(4,*shape)))
                with torch.no_grad():
                    for values in inputs:
                        expected=original(torch.from_numpy(values)).numpy()
                        reference=evaluate_reference(graph,values.tolist())
                        np.testing.assert_allclose(reference,expected,atol=1e-12,rtol=1e-12)
                        np.testing.assert_allclose(native(torch.from_numpy(values)).numpy(),expected,atol=1e-12,rtol=1e-12)
                        np.testing.assert_allclose(user_model(torch.from_numpy(values)).numpy(),expected,atol=1e-12,rtol=1e-12)
                        np.testing.assert_allclose(interpret_fragment(payload,values.reshape(-1)),expected,atol=1e-12,rtol=1e-12)

    def test_suite_is_96_explicit_graphs_and_old_manifest_unchanged(self):
        from catalog_graph_migration import user_graph_suite
        from expanded_model_suite import manifest,FAMILIES,digest
        old=manifest()
        data,origin=user_graph_suite()
        self.assertEqual(manifest(),old)
        self.assertEqual(len(data['cases']),96)
        self.assertTrue(all(c['schema'] in (2,3) and 'family' not in c for c in data['cases']))
        self.assertEqual(len({c['id'] for c in data['cases']}),96)
        self.assertEqual(set(origin['families']),set(FAMILIES))
        self.assertEqual(len(origin['families']),16)
        self.assertFalse(origin['agent_generation_validated'])
        self.assertTrue(origin['historical_reports_unchanged'])
        self.assertEqual(origin['graph_set_sha256'],digest(data['cases']))
        from custom_batch_manifest import load_manifest
        self.assertEqual(load_manifest(BASE/'cases/user-graph-suite-v2.json')[1],data)
        self.assertEqual(load_manifest(BASE/'cases/legacy48-user-graphs.json')[1],dict(schema=1,cases=data['cases'][:48]))
        self.assertEqual(json.loads((BASE/'cases/user-graph-suite-v2.provenance.json').read_text()),origin)
        for case,entry in zip(data['cases'],origin['entries']):
            self.assertEqual(entry['graph_sha256'],digest(case))
            self.assertEqual(entry['source_sha256'],digest(entry['source_descriptor']))

    def test_export_is_not_a_new_family_dispatch_in_the_user_loader(self):
        import numpy as np
        import torch
        from catalog_graph_migration import expand_legacy
        from model_catalog import build_model
        from model_graph import evaluate_reference
        graph=expand_legacy(dict(schema=1,id='linear-1',family='linear',configuration=1))
        values=np.asarray([.5,-1,.25,-.75])
        original=evaluate_reference(graph,values.tolist())
        graph['id']='free-matrix-from-user'
        graph['constants']['weight0'][0][0]+=.173
        graph['nodes'].append(dict(id='extra_negation',op='negate',inputs=[graph['output']]))
        graph['output']='extra_negation'
        expected=evaluate_reference(graph,values.tolist())
        self.assertNotEqual(original,expected)
        model,_=build_model(graph)
        with torch.no_grad(): np.testing.assert_allclose(model(torch.from_numpy(values)).numpy(),expected,atol=1e-12,rtol=1e-12)


@unittest.skipUnless(os.environ.get('POSEIDON_MIGRATED_RULE_REPORT'),'requires actual migrated-rule batch')
class CatalogMigrationEvidenceTests(unittest.TestCase):
    def test_real_exported_graphs_frozen_manifest_and_original_reference(self):
        import numpy as np
        import torch
        from catalog_graph_migration import expand_legacy
        from model_catalog import descriptors,build_model,test_inputs
        from audit_agent_lineage import metadata,read
        from hecate_python_env import WORK
        from seal_cpu_golden import compare
        from seal_artifact_gate import inspect_artifacts
        from hecate_contract import CONTRACT_ROTATIONS
        from cipher_abi import physical_input_names
        path=Path(os.environ['POSEIDON_MIGRATED_RULE_REPORT'])
        report,_=metadata(path,WORK/'results')
        self.assertEqual((report['status'],report['agent_calls']),('passed',0))
        self.assertEqual(report['backend'],'upstream_SEAL_HEVM_CPU')
        self.assertFalse(report['poseidon_gpu_validated'])
        expected=[expand_legacy(old) for old in descriptors()]
        self.assertEqual(report['selected_descriptors'],expected)
        custom=report['custom_manifest']
        snapshot,digest=metadata(path.parent/custom['file'],path.parent)
        self.assertEqual(digest,custom['sha256'])
        self.assertEqual(snapshot,dict(schema=1,cases=expected))
        self.assertEqual(len(report['cases']),48)
        self.assertEqual(report['parameters']['security_check'],'tc128')
        self.assertEqual(report['parameters']['modulus_bits'],[60]*14)
        for row,old,graph in zip(report['cases'],descriptors(),expected):
            self.assertEqual(row['descriptor'],graph)
            self.assertEqual(row['status'],'passed')
            self.assertTrue(row['execution']['encrypted_execution'])
            self.assertFalse(row['execution']['bootstrap_executed'])
            folder=path.parent/row['folder']
            for name,expected_hash in row['frozen_hashes'].items(): self.assertEqual(read(folder/name,folder)[1],expected_hash)
            translation,_=metadata(folder/'translation.json',folder)
            artifact=inspect_artifacts(read(folder/'lowered._hecate_golden.hevm',folder)[0],
                read(folder/'_hecate_golden.cst',folder)[0],
                rotation_steps=CONTRACT_ROTATIONS[translation['static_check']['contract']],
                expected_inputs=len(physical_input_names(translation['layout'])))
            self.assertEqual(artifact,row['artifact_gate'])
            self.assertEqual((row['trace_exit_code'],row['compile_exit_code'],row['execution_exit_code']),(0,0,0))
            self.assertIn('--verify-each',row['compile_command'])
            self.assertEqual(row['execution']['input_batches'],4)
            original,shape=build_model(old)
            with torch.no_grad(): reference=np.stack([original(torch.from_numpy(x)).numpy() for x in test_inputs(shape)])
            with np.load(folder/'arrays.npz',allow_pickle=False) as arrays:
                np.testing.assert_allclose(arrays['reference'],reference,atol=1e-12,rtol=1e-12)
                compared=compare(np.load(folder/'decrypted.npy',allow_pickle=False),arrays['reference'],1e-5,1e-4)
            self.assertEqual(compared,row['comparison'])
        self.assertFalse((path.parent/'private-keys').exists())


if __name__=='__main__':
    unittest.main()
