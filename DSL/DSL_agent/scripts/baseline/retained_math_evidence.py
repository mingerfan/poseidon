"""External provenance verification for the distinct retained-reference format.

This verifies mathematical/reference evidence, never native execution or first
delivery success. A separate real compiler/CKKS witness is always required.
"""
import copy
import math
from pathlib import Path
import struct

from benchmark_graph import validate
from delivery_component import bound,canonical,checked_bound,read_bytes,require,sha
from first_output_backend import _json
from first_output_assessment import MATH,CKKS,compare
import delivery_component
import retained_dual_reference_rows as references
import retained_math_domains as domains


def verify_reference_record(record,selection,*,allow_archived_source_alias=False):
    checked_bound(record);checked_bound(selection)
    require(record.get('format')==references.RECORD and record['provider_selection_binding']==selection['binding']
        and record['target']==selection['target'],'Retained reference selection/target')
    actual=references.freeze(selection['reader_contract'],selection['retained_index'],selection['decoder_source'],selection['target'])
    archived=False
    if actual['provider_source_sha256']!=selection['provider_source_sha256']:
        require(allow_archived_source_alias is True and selection['target']['kind']=='metadata_integration','Archived aliases are retrospective metadata only')
        old=selection['provider_source_sha256'];require(len(old)==len(actual['provider_source_sha256']),'Archived reader source count')
        for path,h in old.items():require(sha(read_bytes(path))==h,'Archived reader source changed')
        for path,h in actual['provider_source_sha256'].items():
            matches=[p for p,v in old.items()if Path(p).name==Path(path).name and v==h]
            require(len(matches)==1,'Archived reader is not byte-identical to the current audited reader')
        payload={k:v for k,v in actual.items()if k!='binding'};payload['provider_source_sha256']=copy.deepcopy(old);actual=bound(payload);archived=True
    require(actual==selection,'Retained source/assets/function selection changed')
    files,_,_,rows,_=references._metadata(selection['reader_contract'],selection['retained_index'],selection['decoder_source'],selection['target'])
    key=record['case_id'];require(key in rows,'Unknown reference case');case,receipt=rows[key]
    require(record['retained_row_receipt']==receipt and record['raw_input_sha256']==receipt['raw_input_sha256']
        and record['declared_input_sha256']==receipt['declared_input_sha256'],'Retained same-case raw/declared identity')
    for key in('native_source_worker_report','model_or_preprocessing_or_DSL_executed','independent_DSL_mathematics_verified',
               'actual_CKKS_verified','application_accuracy_certified','first_output_success_certified'):
        require(record.get(key)is False,'Historical row evidence cannot be promoted')
    require(record.get('reference_origin')=='retained_source_and_declared_rows'
        and record.get('original_source_input_is_reconstruction_not_historical_tensor')is True,'Retained reference origin')
    for role,field in [('original_float32','original_reference'),('approximation','declared_reference')]:
        row=case['references'][role];values=record[field]
        require(type(values)is list and len(values)==selection['target']['output_values']
            and all(type(v)in(int,float)and math.isfinite(v)for v in values),'Finite complete retained references')
        code='<f'if row['dtype']=='<f4'else'<d'
        try:payload=b''.join(struct.pack(code,v)for v in values)
        except (OverflowError,struct.error):raise ValueError('Retained reference dtype changed')from None
        identity=dict(dtype=row['dtype'],bytes=len(payload),sha256=sha(payload))
        require(identity==record['reference_byte_identities'][role]and identity['bytes']==row['bytes']and identity['sha256']==row['row_sha256'],'Retained reference numeric cells changed')
    files.verify()
    return bound(dict(format='poseidon-retained-reference-provenance-result-v1',reference_record_binding=record['binding'],
        provider_selection_binding=selection['binding'],case_id=record['case_id'],archived_reader_alias_verified=archived,
        original_rows_redecoded=0,original_model_or_preprocessing_executed=False,first_output_success_certified=False,native_source_worker_report=False))


def _sources(policy,spec):
    require(policy['independent_math']==MATH and policy['CKKS']==CKKS and spec['producer_sources']==policy['independent_math_sources'],'Original mathematical/CKKS policy and implementation')
    require(all(Path(p).suffix=='.py'and sha(read_bytes(p))==h for p,h in spec['producer_sources'].items())
        and all(Path(p).name!='.env'and sha(read_bytes(p))==h for p,h in spec['producer_assets'].items()),'Frozen mathematical sources/assets changed')
    for p,info in spec['reference_selection']['assets'].items():require(Path(p).name!='.env'and sha(read_bytes(p))==info['sha256'],'Frozen retained asset changed')


def verify_domain_record(descriptor,policy,spec):
    import retained_math_producer as producer
    checked_bound(policy);checked_bound(spec);require(spec==policy['retained_mathematical_producer'],'Selected effective-domain producer');_sources(policy,spec)
    require(type(descriptor)is dict and set(descriptor)=={'path','sha256'}and Path(descriptor['path']).suffix=='.json','Effective-domain evidence descriptor')
    require(sha(read_bytes(descriptor['path']))==descriptor['sha256'],'Effective-domain evidence changed')
    report=checked_bound(_json(descriptor['path']));contract=spec['domain']['effective_declared_domain']
    require(contract is not None and report.get('format')==producer.DOMAIN_RECORD and report.get('request_id')==spec['request_id']
        and report.get('policy_sha256')==sha(canonical(policy))and report.get('producer_spec_binding')==spec['binding']
        and report.get('domain_contract_binding')==contract['binding']and report.get('producer_sources')==spec['producer_sources']
        and report.get('declared_reference_model_sha256')==spec['declared_reference_model_sha256']and report.get('test_only')is spec['test_only'],'Effective-domain model/input/producer scope')
    require(all(report.get(k)is False for k in('candidate_python_executed','CKKS_outputs_used','encrypted_execution','runtime_or_refresh_domain_certified')),'Domain checks are not native evidence')
    model=references._json(read_bytes(spec['declared_model_file']['path']));domains.verify_contract(contract,model);checked=validate(model)
    indices=producer._indices(spec,report['input_indices']);require(len(report['inputs'])==len(indices),'Effective-domain subset count')
    for index,row in zip(indices,report['inputs']):
        binding=spec['input_bindings'][index]
        require(all(row.get(k)==v for k,v in binding.items()),'Effective-domain raw/logical/physical input differs')
        observation=checked_bound(row['domain_observation'])
        require(observation.get('format')==domains.RESULT and observation.get('domain_contract_binding')==contract['binding']
            and observation.get('declared_reference_model_sha256')==spec['declared_reference_model_sha256']
            and observation.get('maximum_scalar_operations')==spec['maximum_scalar_operations']
            and type(observation.get('accounted_scalar_work'))is int and 0<=observation['accounted_scalar_work']<=spec['maximum_scalar_operations'],'Domain operation/model bound')
        require(all(observation.get(k)is False for k in('candidate_python_executed','CKKS_outputs_used','encrypted_execution','runtime_or_refresh_domain_certified')),'Domain native/secret evidence forbidden')
        require(len(observation['observations'])==len(contract['predicates']),'All effective-domain predicates required')
        passed=[]
        for predicate,cell in zip(contract['predicates'],observation['observations']):
            require(cell['predicate_id']==predicate['id']and cell['reference']==predicate['reference']and cell['expected_interval']==predicate['interval'],'Effective domain predicate/bounds changed')
            if cell['all_finite']is True:
                require(type(cell['elements'])is int and cell['elements']==math.prod(checked['shapes'][predicate['reference']])
                    and all(type(cell[k])in(int,float)and math.isfinite(cell[k])for k in('minimum','maximum'))
                    and cell['minimum']<=cell['maximum'],'Actual finite domain extrema/shape')
                ok=predicate['interval'][0]<=cell['minimum']and cell['maximum']<=predicate['interval'][1]
            else:
                require(cell['all_finite']is False and cell['minimum']is None and cell['maximum']is None,'Nonfinite domain cannot claim extrema');ok=False
            require(cell['passed']is bool(ok),'Domain status contradicts observed extrema');passed.append(ok)
        overall=observation['failed_reference_node']is None and all(passed)
        require(observation['passed']is bool(overall),'Domain overall status changed')
    return report


def verify_mathematical_rows(witness,policy,request,source):
    """Only provenance and math comparisons; real native binding stays separate."""
    import retained_math_producer as producer
    checked_bound(policy);spec=checked_bound(policy['retained_mathematical_producer']);_sources(policy,spec)
    require(spec['format']==producer.SPEC and spec['request_id']==request['request_id']and spec['request_sha256']==sha(canonical(request))
        and spec['request_model_sha256']==sha(canonical(request['model']))and witness.get('source_sha256')==sha(source),'Original request/source identity')
    descriptor=witness.get('mathematical_evidence');descriptors=descriptor if type(descriptor)is list else[descriptor]
    require(1<=len(descriptors)<=1024,'Finite mathematical record list')
    root=Path(witness['backend_binding']['evidence']).absolute().parent;by_input={};record_bindings=[]
    fields=('source_sha256','original_reference','declared_reference','dsl_mathematical_output','inside_frozen_domain',
        'domain_sha256','application_metrics','whole_model_output','case_id','raw_input_sha256','declared_input_sha256')
    for item in descriptors:
        require(type(item)is dict and set(item)=={'path','sha256'},'Retained mathematical evidence descriptor')
        path=Path(item['path']).absolute();require(path.suffix=='.json'and path.is_relative_to(root)and path.parent.resolve()==path.parent,'Mathematical evidence outside assessment storage')
        require(sha(read_bytes(path))==item['sha256'],'Retained mathematical record changed')
        report=checked_bound(_json(path))
        require(report.get('format')==producer.RECORD and report.get('request_id')==request['request_id']and report.get('source_sha256')==sha(source)
            and report.get('policy_sha256')==sha(canonical(policy))and report.get('producer_spec_binding')==spec['binding']
            and report.get('producer_sources')==spec['producer_sources']and report.get('original_model_sha256')==policy['original_model_sha256']
            and report.get('declared_model_sha256')==spec['request_model_sha256']and report.get('declared_reference_model_sha256')==spec['declared_reference_model_sha256']
            and report.get('reference_provider_selection_binding')==spec['reference_selection']['binding']
            and report.get('reference_origin')=='retained_source_and_declared_rows'and report.get('test_only')is spec['test_only'],'Retained mathematical model/policy/producer binding')
        require(all(report.get(k)is False for k in('candidate_python_executed','CKKS_outputs_used_as_reference','encrypted_execution','native_source_worker_report','first_output_success_certified')),'Mathematical evidence is not native/source-worker evidence')
        indices=producer._indices(spec,report['input_indices']);require(len(report['inputs'])==len(report['reference_records'])==len(indices),'Retained mathematical input/reference rows')
        effective,status=producer._domain_rows(report['domain_evidence'],policy,spec,indices)
        require(status==report['domain_evidence_status'],'Effective-domain evidence status changed')
        if policy['application_requirements']and report['application_evidence']is None:application={};app_status='missing'
        else:
            application=producer.original._application_values(spec['application_contract'],report['application_evidence'],policy,request,source)
            app_status='verified'if policy['application_requirements']else'not_required'
        require(app_status==report['application_evidence_status'],'Application evidence status changed')
        for index,row,reference in zip(indices,report['inputs'],report['reference_records']):
            verify_reference_record(reference,spec['reference_selection'])
            binding=spec['input_bindings'][index];require(reference['case_id']==binding['case_id']and reference['raw_input_sha256']==binding['raw_input_sha256']
                and reference['declared_input_sha256']==binding['declared_input_sha256'],'Retained mathematical/reference case/input association')
            require(row['input_sha256']==binding['physical_input_sha256']and row['source_sha256']==sha(source)
                and row['retained_reference_record_binding']==reference['binding']and canonical(row['original_reference'])==canonical(reference['original_reference'])
                and canonical(row['declared_reference'])==canonical(reference['declared_reference']),'Original and declared reference values must not be replaced')
            require(type(row['mathematical_scalar_operations'])is int and 0<=row['mathematical_scalar_operations']<=spec['maximum_scalar_operations'],'Independent AST work bound changed')
            require(canonical(row['original_to_declared_approximation'])==canonical(compare(reference['original_reference'],reference['declared_reference'],atol=0.,rtol=0.))
                and canonical(row['independent_math_comparison'])==canonical(compare(reference['declared_reference'],row['dsl_mathematical_output'],**MATH)),'Separate approximation/mathematical observations changed')
            ok=status=='not_required'or status=='verified'and effective[index]['domain_observation']['passed']is True
            require(type(row['input_domain_passed'])is bool and row['effective_domain_status']==status
                and row['inside_frozen_domain']is bool(row['input_domain_passed']and ok)
                and row['domain_sha256']==policy['frozen_domain_sha256']and row['application_evidence_status']==app_status
                and canonical(row['application_metrics'])==canonical(application)and row['whole_model_output']is True,'Domain/application/complete-output evidence differs')
            key=row['input_sha256'];require(key not in by_input,'Duplicate mathematical input record');by_input[key]=row
        record_bindings.append(report['binding'])
    rows=witness.get('inputs');require(type(rows)is list and len(rows)==len(by_input)
        and len({r['input_sha256']for r in rows})==len(rows),'All recorded mathematical input rows must be reported once')
    comparisons=[]
    for row in rows:
        recorded=by_input.get(row['input_sha256'])
        require(recorded is not None and all(canonical(row.get(k))==canonical(recorded.get(k))for k in fields),'Reported cells differ from independent mathematical record')
        comparisons.append(compare(recorded['declared_reference'],recorded['dsl_mathematical_output'],**MATH))
    return bound(dict(format='poseidon-retained-mathematical-provenance-result-v1',record_bindings=record_bindings,input_groups=len(rows),
        all_policy_inputs_covered=set(by_input)==set(policy['inputs']),test_only=spec['test_only'],
        mathematical_compared_values=sum(r['compared_values']for r in comparisons),mathematical_failed_values=sum(r['failed_values']for r in comparisons),
        CKKS_or_compilation_verified_here=False,first_output_success_certified=False))
