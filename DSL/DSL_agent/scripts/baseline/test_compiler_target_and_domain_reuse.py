"""Twelve new controls; synthetic data never constitute a compiler/HE proof."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from delivery_component import bound, canonical, checked_bound, read_bytes, sha
from first_output_assessment import MATH, CKKS
import compiler_target_coordinates as target
import retained_domain_reuse as reuse
import retained_math_domains as domains
import retained_math_producer as producer
import retained_math_evidence as evidence


def rebound(value):
    return bound({k:copy.deepcopy(v) for k,v in value.items() if k != 'binding'})


def compiler_fixture(*, alias=False):
    typ = dict(components=2,kind='ciphertext',level=3,scale_log2=45)
    graph = bound(dict(format=target.DAG,parameters=[dict(ssa='%arg0',value_id='0',type=typ),dict(ssa='%arg1',value_id='1',type=typ)],
        returns=[dict(ssa='%x',value_id='8',type=typ),dict(ssa='%y',value_id='9',type=typ)]))
    segments = [dict(ciphertext=i,value_id=str(8+i),logical_start=0,slot_start=0,elements=n) for i,n in enumerate([3,2])]
    outputs = [dict(name=n,model_value=v,shape=[s['elements']],alias_of=None,segments=[s])
        for n,v,s in zip(['first','second'],['v0','v1'],segments)]
    if alias:
        segments[1]['logical_start'] = 3
        outputs = [dict(name='first',model_value='v0',shape=[5],alias_of=None,segments=segments),
            dict(name='same',model_value='v0',shape=[5],alias_of='first',segments=copy.deepcopy(segments))]
    selectors = [[s['ciphertext'],i] for row in outputs for s in row['segments'] for i in range(s['elements'])]
    layout = bound(dict(format='poseidon-generic-compiler-logical-I-O-map-v1',compiler_DAG_binding=graph['binding'],
        request_id='b'*64,source_sha256='a'*64,
        inputs=[dict(abi_index=i,ssa=r['ssa'],value_id=r['value_id'],role=dict(dsl_name=n,kind=k))
            for i,(r,n,k) in enumerate(zip(graph['parameters'],['x','zero'],['input','encrypted_zero']))],
        outputs=outputs,logical_values=len(selectors),unique_physical_cells=5,alias_cells=len(selectors)-5,
        output_selectors_sha256=sha(canonical(selectors))))
    return bound(dict(format=target.COMPILER,request_id='b'*64,source_sha256='a'*64,DAG=graph,logical_IO=layout,
        periodic_binding=bound(dict(format='poseidon-original-periodic-payload-correspondence-v1',input_slot_period=4,all_non_payload_fields_unchanged=True,plaintext_values_changed=False)),
        original_proof_binding='c'*64,original_backend_identity_sha256='d'*64,original_compiler_sha256='e'*64,
        files_read={'attempt-00/output/lowered._hecate_golden.runtime-plan.json':dict(sha256='f'*64),
            'attempt-00/output/bound/plan.json':dict(sha256='0'*64)}))


def coordinates(value):
    return target.make(value,value['binding'],request_id='b'*64,source_sha256='a'*64)


def domain_fixture(contract=None, *, case_id='case-0', declared_sha='a'*64, shape=None, extrema=None):
    shape = [2] if shape is None else shape
    if contract is None:
        contract = bound(dict(format=domains.FORMAT,declared_reference_model_sha256='b'*64,
            predicates=[dict(id='n0:arg0',reference='input0',interval=[-1.,1.])]))
    extrema = [(-.25,.5)]*len(contract['predicates']) if extrema is None else extrema
    cells = [dict(predicate_id=p['id'],reference=p['reference'],expected_interval=p['interval'],
        elements=2,all_finite=True,minimum=low,maximum=high,passed=p['interval'][0]<=low<=high<=p['interval'][1])
        for p,(low,high) in zip(contract['predicates'],extrema)]
    observation = bound(dict(format=domains.RESULT,domain_contract_binding=contract['binding'],
        declared_reference_model_sha256=contract['declared_reference_model_sha256'],observations=cells,
        passed=all(c['passed'] for c in cells),failed_reference_node=None,maximum_scalar_operations=2**28,accounted_scalar_work=16,
        reference_operation_calls=1,candidate_python_executed=False,CKKS_outputs_used=False,encrypted_execution=False,
        runtime_or_refresh_domain_certified=False))
    logical = dict(case_id=case_id,dtype='<f8',shape=shape,bytes=__import__('math').prod(shape)*8,sha256=declared_sha)
    source = bound(dict(case_id=case_id,dtype='<f8',shape=shape,original_logical_input=logical))
    sources = reuse.implementation_sources()
    ds, ops = [dict(path=str(Path(m.__file__).absolute()),sha256=sources[str(Path(m.__file__).absolute())]) for m in (domains,reuse.benchmark_math)]
    plan = bound(dict(original_logical_input=logical,A_domain_contract=dict(binding=contract['binding']),
        A_declared_domain_source=ds,original_operation_source=ops,source_sha256=sources))
    origin = bound(dict(A_actual_observation_binding=observation['binding'],A_domain_contract_binding=contract['binding'],
        input_record_binding=source['binding'],plan_binding=plan['binding'],case_id=case_id,original_operation_source=ops,
        actually_evaluated_case_count=1,actual_polynomial_argument_observations=len(cells),actual_domain_passed=observation['passed'],
        standalone_development_observation_not_a_pre_generation_policy_record=True,maximum_scalar_operations=2**28,accounted_scalar_work=16))
    metadata = dict(contract=contract,observation=observation,input=source,origin=origin,plan=plan)
    return metadata


def freeze(metadata, **kwargs):
    return reuse.freeze_entry(metadata,{k:v['binding'] for k,v in metadata.items()},test_only=True,**kwargs)


def simple_policy(entry, *, count=44):
    sources = dict(reuse.implementation_sources(), **{str(Path(reuse.__file__).absolute()):sha(read_bytes(reuse.__file__))})
    bindings = [dict(index=i,case_id='case-%d'%i,declared_input_sha256=entry['declared_input']['sha256'] if i==0 else sha(str(i).encode()),
        raw_input_sha256=sha(('raw%d'%i).encode()),logical_input_sha256=sha(('logical%d'%i).encode()),physical_input_sha256=sha(('physical%d'%i).encode())) for i in range(count)]
    spec = bound(dict(request_id='f'*64,test_only=True,producer_sources=sources,input_bindings=bindings,
        declared_reference_model_sha256=entry['domain_observation']['declared_reference_model_sha256'],
        maximum_scalar_operations=2**28,domain=dict(effective_declared_domain=entry['domain_contract'])))
    return bound(dict(independent_math=MATH,CKKS=CKKS,retained_mathematical_producer=spec,inputs=[r['physical_input_sha256'] for r in bindings]))


class TargetAndReuseControls(unittest.TestCase):
    def test_01_compiler_ids_roles_and_all_outputs(self):
        result=coordinates(compiler_fixture())
        self.assertEqual([r['value_id'] for r in result['physical_outputs']],['8','9'])
        self.assertEqual(result['physical_inputs'][1]['role']['kind'],'encrypted_zero')
        self.assertEqual((result['logical_values'],result['unique_physical_cells']),(5,5))
        self.assertFalse(result['program_namespace_assigned']);self.assertFalse(result['mathematical_or_CKKS_or_first_delivery_success_certified'])

    def test_02_exact_aliases_and_tail_chunks_are_counted(self):
        result=coordinates(compiler_fixture(alias=True))
        self.assertEqual((result['logical_values'],result['unique_physical_cells'],result['alias_cells']),(10,5,5))
        self.assertEqual(result['logical_outputs'][1]['segments'][1]['elements'],2)
        self.assertEqual(result['logical_outputs'][1]['alias_of'],'first')

    def test_03_rebound_return_and_chunk_tampering_rejected(self):
        for mutation in ['id','tail','alias']:
            value=compiler_fixture(alias=True)
            if mutation=='id':value['logical_IO']['outputs'][0]['segments'][0]['value_id']='123'
            elif mutation=='tail':value['logical_IO']['outputs'][0]['segments'][1]['elements']=1
            else:value['logical_IO']['outputs'][1]['model_value']='different_expression'
            value['logical_IO']=rebound(value['logical_IO']);value=rebound(value)
            with self.assertRaises(ValueError):coordinates(value)

    def test_04_parameter_order_and_nested_binding_rejected(self):
        for mutation in ['order','dag','source']:
            value=compiler_fixture()
            if mutation=='order':value['logical_IO']['inputs'].reverse()
            elif mutation=='dag':value['logical_IO']['compiler_DAG_binding']='0'*64
            else:value['logical_IO']['source_sha256']='0'*64
            value['logical_IO']=rebound(value['logical_IO']);value=rebound(value)
            with self.assertRaises(ValueError):coordinates(value)

    def test_05_independently_frozen_source_and_receipt_required(self):
        value=compiler_fixture()
        with self.assertRaises(ValueError):target.make(value,'0'*64,request_id='b'*64,source_sha256='a'*64)
        with self.assertRaises(ValueError):target.make(value,value['binding'],request_id='b'*64,source_sha256='0'*64)
        value['DAG']['returns'][1]['value_id']='8';value['DAG']=rebound(value['DAG'])
        value['logical_IO']['compiler_DAG_binding']=value['DAG']['binding'];value['logical_IO']=rebound(value['logical_IO']);value=rebound(value)
        with self.assertRaises(ValueError):coordinates(value)

    def test_06_program_local_origin_and_plan_hashes(self):
        value=coordinates(compiler_fixture());ins=[dict(local_value_id=str(i),logical_value='input%d'%i,chunk_index=0)for i in range(2)]
        outs=[dict(local_value_id=str(8+i),logical_value='terminal%d'%i,chunk_index=0)for i in range(2)]
        source=bound(dict(unit_index=0,inputs=ins,outputs=outs,source_plan_sha256='f'*64,derived_plan_sha256='0'*64))
        unit=dict(unit_id='unit0',source_origin_map_sha256=sha(canonical(source)),raw_plan_sha256='f'*64,plan_sha256='0'*64,
            **{side:[dict(local_value_id=r['local_value_id'],logical=dict(value=r['logical_value'],chunk=r['chunk_index']))for r in rows]for side,rows in [('inputs',ins),('outputs',outs)]})
        result=target.check_program_unit(value,value['binding'],unit,source,unit_index=0)
        self.assertFalse(result['complete_program_target_or_decoder_eligibility_granted'])
        for kind in ['id','plan','origin']:
            changed=copy.deepcopy(unit)
            if kind=='id':changed['outputs'][0]['local_value_id']='0'
            elif kind=='plan':changed['plan_sha256']='1'*64
            else:changed['outputs'][0]['logical']['value']='hand_label'
            with self.assertRaises(ValueError):target.check_program_unit(value,value['binding'],changed,source,unit_index=0)

    def test_07_actual_extrema_and_theorem_annotation_distinct(self):
        meta=domain_fixture();annotation=bound(dict(actual57_observation=dict(binding=meta['observation']['binding']),case_id='case-0',
            claimed_theorem_interval=[-999.,999.]))
        entry=freeze(meta,theorem_annotation=(annotation,annotation['binding']))
        self.assertEqual(entry['domain_observation']['observations'][0]['maximum'],.5)
        self.assertTrue(entry['theorem_annotation']['never_substituted_for_actual_observations'])
        self.assertFalse(entry['new_request_or_policy_bound'])

    def test_08_domain_failure_and_missing_predicate_not_promoted(self):
        entry=freeze(domain_fixture(extrema=[(-.25,2.)]));self.assertFalse(entry['domain_observation']['passed'])
        meta=domain_fixture();meta['observation']['observations']=[];meta['observation']=rebound(meta['observation'])
        meta['origin']['A_actual_observation_binding']=meta['observation']['binding'];meta['origin']=rebound(meta['origin'])
        with self.assertRaises(ValueError):freeze(meta)

    def test_09_new_policy_preserves_complete44_scope(self):
        entry=freeze(domain_fixture());value=reuse.catalog([entry],[entry['binding']]);old=simple_policy(entry)
        snapshot=canonical(old);new=reuse.attach(old,value,value['binding'])
        self.assertEqual(canonical(old),snapshot);self.assertNotEqual(new['binding'],old['binding'])
        self.assertEqual(len(new['inputs']),44);self.assertEqual(len(new['retained_domain_reuse']['missing_required_case_ids']),43)
        self.assertEqual(len(reuse.rows(new,[0])),1)
        with self.assertRaises(ValueError):reuse.rows(new,[1])
        with self.assertRaises(ValueError):reuse.rows(new,list(range(17)))

    def test_10_new_raw_logical_physical_bindings_and_no_case_mix(self):
        entry=freeze(domain_fixture());value=reuse.catalog([entry],[entry['binding']]);policy=simple_policy(entry)
        new=reuse.attach(policy,value,value['binding']);row=reuse.rows(new,[0])[0]
        for key in ['raw_input_sha256','declared_input_sha256','logical_input_sha256','physical_input_sha256']:
            self.assertEqual(row[key],policy['retained_mathematical_producer']['input_bindings'][0][key])
        policy['retained_mathematical_producer']['input_bindings'][0]['declared_input_sha256']='0'*64
        policy['retained_mathematical_producer']=rebound(policy['retained_mathematical_producer']);policy=rebound(policy)
        with self.assertRaises(ValueError):reuse.attach(policy,value,value['binding'])

    def test_11_implementation_and_test_scope_rejected(self):
        meta=domain_fixture();meta['plan']['A_declared_domain_source']['sha256']='0'*64;meta['plan']=rebound(meta['plan'])
        meta['origin']['plan_binding']=meta['plan']['binding'];meta['origin']=rebound(meta['origin'])
        with self.assertRaises(ValueError):freeze(meta)
        entry=freeze(domain_fixture());value=reuse.catalog([entry],[entry['binding']]);policy=simple_policy(entry)
        policy['retained_mathematical_producer']['test_only']=False;policy['retained_mathematical_producer']=rebound(policy['retained_mathematical_producer']);policy=rebound(policy)
        with self.assertRaises(ValueError):reuse.attach(policy,value,value['binding'])
        policy=simple_policy(entry);policy['CKKS']=dict(atol=1.,rtol=1.);policy=rebound(policy)
        with self.assertRaises(ValueError):reuse.attach(policy,value,value['binding'])

    def test_12_reused_domain_record_original_verifier_no_prefix_forward(self):
        # Reuse only the existing synthetic fixture constructor, not its old tests.
        from test_retained_math_producer import Case
        with tempfile.TemporaryDirectory() as folder:
            case=Case(Path(folder)/'case',composed=True)
            policy=case.policy(extra_sources=(str(Path(reuse.__file__).absolute()),))
            spec=policy['retained_mathematical_producer'];identity=spec['input_bindings'][0]
            values=case.samples[0]['input0'][0][0][0]
            extrema=[(min(values),max(values)),(2*min(values),2*max(values))]
            meta=domain_fixture(case.domain_contract,case_id='case-0',declared_sha=identity['declared_input_sha256'],shape=[1,1,1,2],extrema=extrema)
            entry=freeze(meta);value=reuse.catalog([entry],[entry['binding']]);policy=reuse.attach(policy,value,value['binding'])
            with patch.object(domains,'evaluate',side_effect=AssertionError('No retained domain prefix replay')):
                descriptor=reuse.produce(case.request,case.selection,case.samples,case.raw,policy,Path(folder)/'reused.json',indices=[0])
            record=evidence.verify_domain_record(descriptor,policy,policy['retained_mathematical_producer'])
            self.assertEqual(record['new_domain_prefix_executions'],0);self.assertEqual(record['inputs'][0]['domain_observation'],meta['observation'])
            self.assertFalse(record['historical_request_policy_generation_or_observation_modified'])
            changed=json.loads(read_bytes(descriptor['path']));changed['inputs'][0]['domain_observation']['observations'][0]['elements']=3
            changed['inputs'][0]['domain_observation']=rebound(changed['inputs'][0]['domain_observation']);changed=rebound(changed)
            other=Path(folder)/'badshape.json';other.write_bytes(canonical(changed))
            with self.assertRaises(ValueError):evidence.verify_domain_record(dict(path=str(other),sha256=sha(read_bytes(other))),policy,policy['retained_mathematical_producer'])


if __name__ == '__main__':unittest.main()
