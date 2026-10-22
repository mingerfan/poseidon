
"""Actual upstream calls on public vectors; no encrypted-success claims."""
import copy,unittest
import numpy as np
from benchmark_suite import generate,Builder
from benchmark_graph import samples,digest
from benchmark_math import evaluate
from benchmark_torch import evaluate as torch_reference
from unified_graph_contract import layout
from test_upstream_spatial import Vector
from upstream_adapters import spatial_mapped

from upstream_spatial_mapped_cases import corpus,asymmetric

class MappedVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from upstream_adapters.test_batch_norm import BatchNormAdapterTests
        BatchNormAdapterTests.setUpClass.__func__(cls)
    def test_frozen_models_and_asymmetric_parameters(self):
        for model in corpus()+asymmetric():
            p=layout(model)['input_slot_period']
            with self.subTest(model=model['id']):
                bindings={n['id']:spatial_mapped.bind_node(model,n['id'],p) for n in model['nodes'] if n['op']!='negate'}
                for inputs in samples(model,16):
                    expected=evaluate(model,inputs);reference=torch_reference(model,inputs)
                    env={}
                    for entry in model['inputs']:
                        flat=inputs[entry['name']].reshape(-1)
                        env[entry['name']]=Vector(np.tile(np.pad(flat,(0,p-len(flat))),16384//p))
                    for node in model['nodes']:
                        value=env[node['inputs'][0]]
                        if node['op']=='negate':out=Vector(-value.data)
                        else:
                            binding=bindings[node['id']]
                            out,record=spatial_mapped.apply(binding,value,self.helpers,self.mpcb)
                            spatial_mapped.verify_record(binding,record)
                            if binding.get('adapter')==spatial_mapped.ADAPTER:
                                probe=spatial_mapped.probe(binding,tuple(value.data[:p]))
                                np.testing.assert_allclose(out.data[:p],probe,atol=1e-12,rtol=1e-12)
                        env[node['outputs'][0]]=out
                    for output in model['outputs']:
                        ref=expected[output['name']].reshape(-1);actual=env[output['value']].data
                        np.testing.assert_allclose(ref,reference[output['name']].reshape(-1),atol=1e-12,rtol=1e-12)
                        np.testing.assert_allclose(actual[:len(ref)],ref,atol=1e-12,rtol=1e-12)
                        np.testing.assert_array_equal(actual,np.tile(actual[:p],16384//p))


    def test_actual_record_rejects_missing_calls_rotations_and_work(self):
        model=corpus()[0];p=layout(model)['input_slot_period']
        binding=spatial_mapped.bind_node(model,model['nodes'][0]['id'],p)
        vec=Vector(np.tile(np.arange(p)/32,16384//p))
        _,actual=spatial_mapped.apply(binding,vec,self.helpers,self.mpcb)
        for kind in ('calls','rotation','work','helper','bootstrap'):
            bad=copy.deepcopy(actual)
            if kind=='calls':bad['inner_calls']=[]
            if kind=='rotation':bad['mapping_rotations'][0]['normalized']+=1
            if kind=='work':bad['mapping_operations']+=1
            if kind=='helper':bad['inner_calls'][0]['helper']='HE_Pool'
            if kind=='bootstrap':bad['bootstrap_removed']=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):spatial_mapped.verify_record(binding,bad)

class MappedContractTests(unittest.TestCase):
    def request(self,row):
        from unified_graph_contract import prepare
        from compiler_configuration import PROFILE_SHA256,configuration
        return prepare(row['model'],PROFILE_SHA256,configuration(row['configuration']),
            helper_profile=row['profile'],helper_exercise=row['required_helpers'])
    def test_all_corpus_and_asymmetric_programs_fit_unchanged_budget(self):
        from upstream_spatial_mapped_cases import cases
        from unified_graph_contract import validate_candidate
        for row in cases():
            with self.subTest(name=row['name']):
                request=self.request(row)
                checked=validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=row['source']),request)
                self.assertEqual(set(checked['upstream_exercise']['witnesses']),set(row['required_helpers']))
    def test_rehashed_maps_weights_and_call_count_rejected(self):
        from upstream_spatial_mapped_cases import cases
        from unified_graph_contract import validate_request
        request=self.request(cases()[0])
        for kind in ('map','coefficient','weight','count','work'):
            bad=copy.deepcopy(request);binding=bad['upstream_helpers']['helpers']['HE_Conv0']['binding']
            if kind=='map':binding['calls'][0]['before'][0]['step']+=1
            if kind=='coefficient':binding['calls'][0]['after'][0]['mask'][0]+=1
            if kind=='weight':binding['calls'][0]['inner']['parameters']['weight'][0][0][0][0]+=.1
            if kind=='count':binding['calls'].append(copy.deepcopy(binding['calls'][0]))
            if kind=='work':binding['work']=1
            bad['request_id']=digest({k:v for k,v in bad.items() if k!='request_id'})
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_request(bad)
    def test_discarding_actual_mapped_helper_is_not_contribution(self):
        from upstream_spatial_mapped_cases import cases
        from unified_graph_contract import validate_candidate
        row=cases()[0];request=self.request(row)
        source='@hc.func("c,c")'+chr(10)+'def golden(x,zero_ct):'+chr(10)+'    h = HE_Conv0(x)'+chr(10)+'    return h-h+x'+chr(10)
        with self.assertRaisesRegex(ValueError,'contribution'):
            validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=source),request)
    def test_v4_retains_original_rejections(self):
        from upstream_candidate_helpers import manifest,SPATIAL_PROFILE,MAPPED_PROFILE
        g=corpus()[0]
        old=manifest(g,SPATIAL_PROFILE);new=manifest(g,MAPPED_PROFILE)
        self.assertNotIn('HE_Conv0',old['helpers']);self.assertIn('HE_Conv0',new['helpers'])
        self.assertTrue(old['unavailable_bindings']);self.assertFalse(new['unavailable_bindings'])
    def test_resources_effective_kernel_and_axes_remain_bounded(self):
        b=Builder([(1,1,15,15)])
        y=b.node('avg_pool2d',['input0'],kernel=[2,2],stride=[1,1],padding=[0,0],count_include_pad=False)
        g=b.finish(y)
        with self.assertRaisesRegex(ValueError,'budget'):spatial_mapped.bind_node(g,g['nodes'][0]['id'],256)
        b=Builder([(1,1,3,3)])
        y=b.node('mean',['input0'],axes=[1,2],keepdims=True);g=b.finish(y)
        with self.assertRaisesRegex(ValueError,'spatial axes'):spatial_mapped.bind_node(g,g['nodes'][0]['id'],16)
    def test_v5_is_opt_in_and_chunk_mixing_rejected(self):
        from upstream_spatial_mapped_cases import cases
        from unified_graph_contract import prepare
        from compiler_configuration import PROFILE_SHA256
        g=cases()[0]['model']
        self.assertNotIn('upstream_helpers',prepare(g,PROFILE_SHA256))
        with self.assertRaises(ValueError):prepare(g,PROFILE_SHA256,helper_profile='upstream-poly-spatial-mapped-v5',chunk_period=4)

if __name__=='__main__':unittest.main()
