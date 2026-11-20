"""Public catalog, provenance compatibility and rejection-boundary regressions."""
import hashlib,json,unittest
from pathlib import Path
from unittest.mock import patch
from types import MappingProxyType
from model_decomposition import REGISTRY,decompose,FORMAT as GRAPH_FORMAT
from operator_registry_contracts import catalog,SPECS,FORMAT

BASE=Path(__file__).parent
GOLDEN=BASE/'benchmarks/tools/current_registry_migration_v1/r1807/bindings-before.json'

class OperatorRegistryContracts(unittest.TestCase):
    def test_catalog_covers_every_registered_attribute(self):
        value=catalog()
        self.assertEqual(value['format'],FORMAT)
        self.assertEqual(len(value['operators']),34)
        self.assertEqual(set(value['operators']),set(REGISTRY))
        for name,entry in value['operators'].items():
            self.assertEqual(set(entry['public_attributes']),set(REGISTRY[name].attributes))
            self.assertEqual(entry['reference'],REGISTRY[name].reference)
            self.assertEqual(entry['decomposition_rule'],REGISTRY[name].rule)
            self.assertEqual(entry['input_dtype'],'float64')
            self.assertEqual(entry['value_type'],'real_tensor')
            self.assertEqual(entry['graph_format'],GRAPH_FORMAT)
            self.assertEqual(entry['input_roles'][0],'cipher_tensor')
            self.assertTrue(entry['shape_constraints']and entry['output_shape'])
            self.assertNotEqual(entry['summary'],'checked by rule and core shape inference')
    def test_catalog_immutable_and_returned_data_independent(self):
        with self.assertRaises(TypeError):SPECS['unknown']=None
        value=catalog();value['operators']['matmul']['input_roles'].append('forged')
        self.assertEqual(catalog()['operators']['matmul']['input_roles'],['cipher_tensor','cipher_tensor'])
    def test_registry_extension_requires_corresponding_contract(self):
        with patch('model_decomposition.REGISTRY',MappingProxyType({})):
            with self.assertRaisesRegex(ValueError,'catalog coverage'):catalog()
    def test_public_capabilities_expose_contract_without_execution_claims(self):
        from application_capabilities import describe
        from benchmark_graph import digest
        value=describe();registry=value['decomposition_registry']
        self.assertEqual(registry['contract_format'],FORMAT)
        for name,entry in registry['operators'].items():
            self.assertEqual(entry['input_contract'],catalog()['operators'][name]['summary'])
            self.assertEqual(entry['type_and_shape'],catalog()['operators'][name])
            self.assertEqual(entry['compiler_status'],'unverified')
            self.assertEqual(entry['encrypted_status'],'unverified')
            self.assertEqual(entry['agent_status'],'unverified')
        self.assertEqual(value['identity_sha256'],digest({k:v for k,v in value.items()if k!='identity_sha256'}))
    def test_source_closure_tracks_catalog_module(self):
        from application_identity import source_closure
        value=source_closure(BASE,('application_capabilities.py',))
        self.assertIn('operator_registry_contracts.py',value['sources'])
        self.assertEqual(value['sources']['operator_registry_contracts.py'],hashlib.sha256((BASE/'operator_registry_contracts.py').read_bytes()).hexdigest())
    def test_all_migrated_models_and_bindings_are_byte_equal(self):
        from benchmark_graph import canonical
        saved=json.loads(GOLDEN.read_text());self.assertEqual(len(saved['cases']),45)
        self.assertEqual(hashlib.sha256((BASE/'model_decomposition.py').read_bytes()).hexdigest(),saved['decomposition_sha256'])
        for row in saved['cases']:
            source=Path(row['source']);self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),row['source_sha256'])
            original=json.loads(source.read_text());lowered,binding=decompose(original)
            self.assertEqual(canonical(lowered).hex(),row['lowered_canonical_hex'])
            self.assertEqual(canonical(binding).hex(),row['binding_canonical_hex'])
    def test_catalog_does_not_widen_legacy_public_dot(self):
        original=dict(format=GRAPH_FORMAT,id='legacy_public_dot',inputs=[dict(name='x',shape=[2])],constants=dict(public=[.25,.5]),nodes=[dict(id='dot',op='dot',inputs=['x','public'],attrs={},outputs=['y'])],outputs=[dict(name='result',value='y')])
        with self.assertRaisesRegex(ValueError,'Encrypted dot/matmul operands'):decompose(original)
    def test_catalog_does_not_widen_public_gather_indices(self):
        original=dict(format=GRAPH_FORMAT,id='negative_gather',inputs=[dict(name='x',shape=[2])],constants={},nodes=[dict(id='gather',op='public_gather',inputs=['x'],attrs=dict(axis=0,indices=[-1]),outputs=['y'])],outputs=[dict(name='result',value='y')])
        with self.assertRaisesRegex(ValueError,'Public gather indices'):decompose(original)

if __name__=='__main__':unittest.main()
