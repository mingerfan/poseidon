"""Twelve new synthetic controls; no real model/native/acceptance execution."""
import copy
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from component_contract import prepare_task
from candidate_contract import request_input_names
from delivery_component import bound,canonical,read_bytes,sha
from first_output_assessment import evaluate_witness
from first_output_dsl_math import MathematicalWorkLimit
from test_retained_dual_reference_rows import Fixture
import retained_dual_reference_rows as references
import retained_math_domains as domains
import retained_math_producer as producer
import retained_math_evidence as evidence


class Case:
    def __init__(self,root,*,composed=False,outside=False):
        self.root=root;self.fixture=f=Fixture(root)
        model=dict(format='poseidon-model-graph-v1',id='retained_math_fixture',inputs=[dict(name='input0',shape=[1,1,1,2])],
            constants={},nodes=[dict(id='n0',op='square',inputs=['input0'],attrs={},outputs=['v0'])],outputs=[dict(name='output0',value='v0')])
        if composed:
            model['constants']=dict(c0=[0.,2.],c1=[0.,.5])
            model['nodes']=[dict(id='n0',op='polynomial',inputs=['input0','c0'],attrs=dict(basis='power'),outputs=['v0']),
                dict(id='n1',op='polynomial',inputs=['v0','c1'],attrs=dict(basis='power'),outputs=['v1'])]
            model['outputs'][0]['value']='v1'
        self.model=model;self.request=prepare_task(model)
        original=copy.deepcopy(f.original_value);relation=copy.deepcopy(f.relation_value)
        provenance=copy.deepcopy(f.provenance)
        p=Path(provenance['full_ModelSpec']['path']);p.write_bytes(canonical(model)+b'\n')
        provenance['full_ModelSpec'].update(sha256=sha(p.read_bytes()),bytes=p.stat().st_size)
        p=Path(provenance['approximation_definition']['path']);p.write_bytes(canonical(dict(domains=dict(unit=[-1.,1.])))+b'\n')
        provenance['approximation_definition'].update(sha256=sha(p.read_bytes()),bytes=p.stat().st_size)
        if outside:
            raw,wire=f.inputs['case-0'];values=list(struct.unpack('<dd',wire));values[0]=.75;f.inputs['case-0']=(raw,struct.pack('<dd',*values))
        declared=[]
        for key in ['case-0','case-1']:
            values=list(struct.unpack('<dd',f.inputs[key][1]));declared.append(values if composed else[v*v for v in values])
        arrays=copy.deepcopy(f.arrays)
        for name,data,dtype in [('declared64',declared,'<f8'),('source32',np.asarray(declared,dtype='<f4')+np.float32(.01),'<f4')]:
            p=Path(arrays[name]['path']);np.save(p,np.asarray(data,dtype=dtype),allow_pickle=False);raw=p.read_bytes()
            arrays[name].update(sha256=sha(raw),bytes=len(raw))
        original['public_files']=dict(provenance,**arrays);original['original_full_test']=dict(images=10000)
        for i,case in enumerate(original['cases']):
            key=case['case_id'];case['normalized_logical_f64_sha256']=sha(f.inputs[key][1])
            for row in case['references'].values():
                desc=arrays[row['source_key']];raw=Path(desc['path']).read_bytes();row['row_sha256']=sha(raw[row['file_byte_offset']:row['file_byte_offset']+row['bytes']])
        clean=lambda v:{k:x for k,x in v.items()if k!='binding'}
        original_desc,original=f.document('original.json',clean(original))
        relation['source_graph_weights_and_approximation']=provenance
        for i,link in enumerate(relation['cases']):
            link['normalized_logical_float64_sha256']=original['cases'][i]['normalized_logical_f64_sha256']
            link['retained_original_and_declared_reference_rows']=copy.deepcopy(original['cases'][i]['references'])
        relation_desc,relation=f.document('relation.json',clean(relation))
        contract=copy.deepcopy(f.contract_value);contract.update(original_contract=original_desc,input_relation=relation_desc)
        contract_desc,contract=f.document('contract.json',clean(contract))
        index=copy.deepcopy(f.index_value);index['reader_contract']=contract_desc
        for i,receipt in enumerate(index['cases']):
            receipt.update(original_contract=original_desc,input_relation=relation_desc,reader_contract_binding=contract['binding'],
                declared_input_sha256=original['cases'][i]['normalized_logical_f64_sha256'],references=copy.deepcopy(original['cases'][i]['references']))
            index['cases'][i]=bound(clean(receipt))
        index_desc,_=f.document('index.json',clean(index))
        self.target=dict(f.target,kind='frozen_generation_request',request_id=self.request['request_id'],declared_model_file_sha256=provenance['full_ModelSpec']['sha256'])
        self.selection=references.freeze(contract_desc,index_desc,f.decoder,self.target)
        self.samples=[dict(input0=np.frombuffer(f.inputs[k][1],dtype='<f8').reshape(1,1,1,2).tolist())for k in ['case-0','case-1']]
        self.raw=[dict(case_id=k,raw_input=f.inputs[k][0])for k in ['case-0','case-1']]
        names=request_input_names(self.request);body=names[0]if composed else names[0]+'*'+names[0]
        self.source=('@hc.func("'+','.join('c'for _ in names)+'")\r\ndef golden('+', '.join(names)+'):\r\n    return '+body+'\r\n\r\n').encode()
        self.domain_contract=domains.contract(model,provenance['approximation_definition']if composed else None,
            {n['id']:['domains','unit']for n in model['nodes']if n['op']=='polynomial'})
        self.number=0
    def path(self,name):
        self.number+=1;return self.root/(str(self.number)+'-'+name+'.json')
    def policy(self,**kwargs):
        defaults=dict(domain=dict(input0=[-1.,1.]),approximation_requirements=[],application_requirements=[],test_only=True,
            domain_contract=self.domain_contract if self.model['constants']else None)
        defaults.update(kwargs)
        return producer.make_policy(self.request,self.selection,self.samples,self.raw,**defaults)
    def domain(self,policy,**kwargs):
        return producer.produce_domain_record(self.request,self.selection,self.samples,self.raw,policy,self.path('domain'),**kwargs)
    def record(self,policy=None,source=None,**kwargs):
        policy=self.policy()if policy is None else policy;source=self.source if source is None else source
        descriptor=producer.produce_record(self.request,source,self.selection,self.samples,self.raw,policy,self.path('math'),**kwargs)
        return __import__('json').loads(read_bytes(descriptor['path'])),descriptor
    def witness(self,record,descriptor,source=None):
        return dict(source_sha256=sha(self.source if source is None else source),backend_binding=dict(evidence=str(self.root/'native')),
            mathematical_evidence=descriptor,inputs=copy.deepcopy(record['inputs']))


class RetainedMathematicalControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bundle=producer.implementation_bundle()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.n=0
        def cached(extra_sources=()):
            bundle=copy.deepcopy(self.bundle);bundle['sources'].update({str(Path(p)):sha(read_bytes(p))for p in extra_sources});return bundle
        cache=patch.object(producer,'implementation_bundle',side_effect=cached);cache.start();self.addCleanup(cache.stop)
    def case(self,**kwargs):self.n+=1;return Case(self.root/str(self.n),**kwargs)
    def application(self):
        p=next(iter(self.bundle['sources']));return dict(samples=10000,dataset_sha256='e'*64,producer_sources={p:self.bundle['sources'][p]})
    def whole_policy(self,c):
        return c.policy(whole_model_required=True,domain_contract=c.domain_contract,
            application_requirements=[dict(id='top1_accuracy_difference',comparison='<=',limit=.01)],application_contract=self.application(),required_application_samples=10000)
    def app_report(self,c,policy,*,samples=10000,source_sha=None):
        contract=policy['retained_mathematical_producer']['application_contract']
        report=bound(dict(format='poseidon-first-output-application-evidence-v1',contract_sha256=sha(canonical(contract)),
            original_model_sha256=policy['original_model_sha256'],declared_model_sha256=sha(canonical(c.request['model'])),
            source_sha256=sha(c.source)if source_sha is None else source_sha,samples=samples,dataset_sha256=contract['dataset_sha256'],
            producer_sources=contract['producer_sources'],candidate_python_executed=False,CKKS_outputs_used_as_reference=False,
            metrics=dict(top1_accuracy_difference=.005)))
        p=c.path('synthetic-application');p.write_bytes(canonical(report));return dict(path=str(p),sha256=sha(read_bytes(p)))

    def test_policy_binds_raw_declared_logical_physical_and_function_roles(self):
        c=self.case();policy=c.policy();spec=policy['retained_mathematical_producer']
        self.assertEqual(policy['independent_math'],dict(atol=1e-12,rtol=1e-12));self.assertEqual(policy['CKKS'],dict(atol=1e-5,rtol=1e-4))
        self.assertEqual(spec['reference_kind'],'retained_source_and_declared_rows');self.assertTrue(spec['test_only'])
        for row in spec['input_bindings']:
            self.assertTrue(all(len(row[k])==64 for k in ['raw_input_sha256','declared_input_sha256','logical_input_sha256','physical_input_sha256']))
        self.assertNotEqual(spec['original_model_sha256'],spec['declared_reference_model_sha256'])

    def test_correct_AST_math_keeps_original_and_declared_values_and_source_bytes(self):
        c=self.case();policy=c.policy();record,descriptor=c.record(policy)
        self.assertNotEqual(record['inputs'][0]['original_reference'],record['inputs'][0]['declared_reference'])
        self.assertEqual(record['inputs'][0]['declared_reference'],record['inputs'][0]['dsl_mathematical_output'])
        self.assertEqual(record['source_sha256'],sha(c.source));self.assertEqual(Path(descriptor['path']).stat().st_mode&0o777,0o400)
        result=evidence.verify_mathematical_rows(c.witness(record,descriptor),policy,c.request,c.source)
        self.assertEqual(result['mathematical_failed_values'],0);self.assertTrue(result['test_only']);self.assertFalse(result['CKKS_or_compilation_verified_here'])

    def test_wrong_legal_DSL_is_observed_without_repair_or_reference_substitution(self):
        c=self.case();policy=c.policy();name=request_input_names(c.request)[0];wrong=c.source.replace((name+'*'+name).encode(),name.encode())
        record,descriptor=c.record(policy,source=wrong);result=evidence.verify_mathematical_rows(c.witness(record,descriptor,wrong),policy,c.request,wrong)
        self.assertGreater(result['mathematical_failed_values'],0);self.assertEqual(record['source_sha256'],sha(wrong));self.assertFalse(record['first_output_success_certified'])

    def test_changed_private_inputs_model_or_physical_binding_are_rejected(self):
        c=self.case();policy=c.policy();raw=copy.deepcopy(c.raw);raw[0]['raw_input']=b'xx'
        with self.assertRaises(ValueError):producer.check_policy(policy,c.request,c.selection,c.samples,raw)
        samples=copy.deepcopy(c.samples);samples[0]['input0'][0][0][0][0]+=.125
        with self.assertRaises(ValueError):producer.check_policy(policy,c.request,c.selection,samples,c.raw)
        changed=copy.deepcopy(c.request);changed['model']['id']='changed'
        with self.assertRaises(ValueError):producer.check_policy(policy,changed,c.selection,c.samples,c.raw)
        changed=copy.deepcopy(policy);changed.pop('binding');spec=changed['retained_mathematical_producer'];spec.pop('binding');spec['input_bindings'][0]['physical_input_sha256']='d'*64
        changed['retained_mathematical_producer']=bound(spec);changed=bound(changed)
        with self.assertRaises(ValueError):producer.check_policy(changed,c.request,c.selection,c.samples,c.raw)

    def test_complete_case_set_and_original_application_count_cannot_be_shortened(self):
        c=self.case()
        with self.assertRaises(ValueError):c.policy(whole_model_required=True)
        with self.assertRaises(ValueError):c.policy(whole_model_required=True,domain_contract=c.domain_contract,
            application_requirements=[dict(id='top1_accuracy_difference',comparison='<=',limit=.01)],application_contract=dict(self.application(),samples=9999),required_application_samples=9999)
        policy=self.whole_policy(c);self.assertEqual(policy['retained_mathematical_producer']['required_application_samples'],10000)
        with self.assertRaises(ValueError):producer.make_policy(c.request,c.selection,c.samples[:1],c.raw[:1],domain=dict(input0=[-1.,1.]),
            approximation_requirements=[],application_requirements=[dict(id='top1_accuracy_difference',comparison='<=',limit=.01)],
            application_contract=self.application(),required_application_samples=10000,domain_contract=c.domain_contract,whole_model_required=True,test_only=True)

    def test_missing_domain_and_application_evidence_stay_unverified(self):
        c=self.case();policy=self.whole_policy(c);record,descriptor=c.record(policy)
        self.assertEqual(record['domain_evidence_status'],'missing');self.assertEqual(record['application_evidence_status'],'missing')
        self.assertTrue(all(not row['inside_frozen_domain']and row['application_metrics']=={}for row in record['inputs']))
        evidence.verify_mathematical_rows(c.witness(record,descriptor),policy,c.request,c.source)
        witness=dict(c.witness(record,descriptor),compiled_validated=True,real_compiler=True,compiler_source_sha256=sha(c.source),test_only=True)
        for row in witness['inputs']:row['CKKS_real']=row['declared_reference'];row['CKKS_imag']=None
        verdict=evaluate_witness(witness,policy,sha(c.source));self.assertFalse(verdict['raw_numeric_passed'])
        self.assertTrue(all(not row['domain_passed']and not row['application_passed']for row in verdict['inputs']))

    def test_actual_internal_domain_violation_is_not_hidden_by_valid_inputs_or_outputs(self):
        c=self.case(composed=True,outside=True);policy=c.policy();domain=c.domain(policy)
        record,descriptor=c.record(policy,domain_evidence=domain);row=record['inputs'][0]
        self.assertTrue(row['input_domain_passed']);self.assertFalse(row['inside_frozen_domain']);self.assertEqual(row['declared_reference'],row['dsl_mathematical_output'])
        observations=__import__('json').loads(read_bytes(domain['path']))['inputs'][0]['domain_observation']['observations']
        self.assertEqual(observations[1]['maximum'],1.5);self.assertFalse(observations[1]['passed'])
        evidence.verify_mathematical_rows(c.witness(record,descriptor),policy,c.request,c.source)

    def test_domain_coverage_definition_and_bounds_are_frozen(self):
        c=self.case(composed=True)
        with self.assertRaises(ValueError):domains.contract(c.model,c.domain_contract['definition'],dict(n0=['domains','unit']))
        changed=copy.deepcopy(c.domain_contract);changed.pop('binding');changed['predicates'][0]['interval']=[-9.,9.]
        with self.assertRaises(ValueError):domains.verify_contract(bound(changed),c.model)
        with self.assertRaises(ValueError):c.policy(domain_contract=None)
        different=copy.deepcopy(c.domain_contract);different.pop('binding');different['definition']=dict(different['definition'],sha256='a'*64)
        with self.assertRaises(ValueError):c.policy(domain_contract=bound(different))

    def test_application_dataset_and_source_provenance_require_the_full_frozen_report(self):
        c=self.case();policy=self.whole_policy(c);domain=c.domain(policy)
        for descriptor in [self.app_report(c,policy,samples=9999),self.app_report(c,policy,source_sha='b'*64)]:
            with self.assertRaises(ValueError):c.record(policy,domain_evidence=domain,application_evidence=descriptor)
        record,descriptor=c.record(policy,domain_evidence=domain,application_evidence=self.app_report(c,policy))
        self.assertEqual(record['application_evidence_status'],'verified');self.assertTrue(all(r['inside_frozen_domain']for r in record['inputs']))
        evidence.verify_mathematical_rows(c.witness(record,descriptor),policy,c.request,c.source)

    def test_original_scalar_limits_and_no_retry_record_paths_are_enforced(self):
        c=self.case();policy=c.policy(maximum_scalar_operations=1);path=c.path('limited')
        with self.assertRaises(MathematicalWorkLimit):producer.produce_record(c.request,c.source,c.selection,c.samples,c.raw,policy,path)
        self.assertTrue(path.with_name(path.name+'.attempt.json').exists())
        with self.assertRaises(FileExistsError):producer.produce_record(c.request,c.source,c.selection,c.samples,c.raw,policy,path)
        with self.assertRaises(ValueError):c.policy(maximum_scalar_operations=2**28+1)
        c=self.case(composed=True);policy=c.policy(maximum_scalar_operations=1)
        with self.assertRaises(domains.DomainWorkLimit):c.domain(policy)

    def test_tampered_reported_cells_retained_reference_or_domain_status_rejected(self):
        c=self.case();policy=c.policy();record,descriptor=c.record(policy);witness=c.witness(record,descriptor)
        witness['inputs'][0]['dsl_mathematical_output'][0]+=1.
        with self.assertRaises(ValueError):evidence.verify_mathematical_rows(witness,policy,c.request,c.source)
        forged=copy.deepcopy(record);forged.pop('binding');reference=forged['reference_records'][0];reference.pop('binding');reference['original_reference'][0]+=.25
        reference=bound(reference);forged['reference_records'][0]=reference;forged['inputs'][0]['original_reference']=reference['original_reference'];forged['inputs'][0]['retained_reference_record_binding']=reference['binding'];forged=bound(forged)
        p=c.path('forged-reference');p.write_bytes(canonical(forged));desc=dict(path=str(p),sha256=sha(read_bytes(p)))
        with self.assertRaises(ValueError):evidence.verify_mathematical_rows(c.witness(forged,desc),policy,c.request,c.source)
        c=self.case(composed=True,outside=True);policy=c.policy();descriptor=c.domain(policy);report=__import__('json').loads(read_bytes(descriptor['path']));report.pop('binding')
        observation=report['inputs'][0]['domain_observation'];observation.pop('binding');observation['observations'][1]['passed']=True;observation['passed']=True
        report['inputs'][0]['domain_observation']=bound(observation);report=bound(report);p=c.path('forged-domain');p.write_bytes(canonical(report))
        with self.assertRaises(ValueError):evidence.verify_domain_record(dict(path=str(p),sha256=sha(read_bytes(p))),policy,policy['retained_mathematical_producer'])

    def test_full_planned_count_uses_original_chunks_and_partial_records_cannot_be_full(self):
        c=self.case();logical,physical,wire=producer._normalized(c.request,c.samples*22);self.assertEqual(len(logical),44);self.assertEqual(len(physical),44)
        with self.assertRaises(ValueError):producer._indices(dict(input_bindings=[{}]*44),list(range(17)))
        policy=c.policy();a,da=c.record(policy,indices=[0]);b,db=c.record(policy,indices=[1])
        result=evidence.verify_mathematical_rows(c.witness(a,da),policy,c.request,c.source);self.assertFalse(result['all_policy_inputs_covered'])
        witness=c.witness(a,[da,db]);witness['inputs']+=b['inputs'];result=evidence.verify_mathematical_rows(witness,policy,c.request,c.source)
        self.assertTrue(result['all_policy_inputs_covered']);self.assertFalse(result['first_output_success_certified'])
        from first_output_numerical_evidence import verify_mathematical_rows as legacy
        with self.assertRaises(ValueError):legacy(c.witness(a,da),policy,c.request,c.source)


if __name__=='__main__':unittest.main()
