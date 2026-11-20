"""Read-only intake of independently frozen library and key-generation evidence.

All inputs are public JSON bytes. No path is opened, no executable is loaded,
and no private key, private-key hash, backend or execution grant is returned.
The result is provenance for a new SDK, not a replacement for its runtime
capability, artifact attestation or actual encrypted acceptance.
"""
import copy
import hashlib
import json
import math
import re

DOCUMENTS=frozenset({'approval','library_plan','key_plan','library_report',
    'key_report','public_origin','public_manifest','library_terminal','key_terminal'})
SHA=re.compile(r'[0-9a-f]{64}\Z')
EPOCH=re.compile(r'[0-9a-f]{32}\Z')
MAX_DOCUMENT=1024**2
MAX_TOTAL=2*1024**2


def need(condition,reason):
    if not condition:raise ValueError('Native key origin intake: '+reason)


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,
                      separators=(',',':'),allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def _json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            need(key not in result,'duplicate JSON key');result[key]=value
        return result
    def real(text):
        value=float(text);need(math.isfinite(value),'nonfinite JSON');return value
    def invalid(_):need(False,'nonfinite JSON')
    need(type(raw)is bytes and 0<len(raw)<=MAX_DOCUMENT,'bounded public JSON')
    try:return json.loads(raw,object_pairs_hook=pairs,parse_float=real,parse_constant=invalid)
    except (RecursionError,UnicodeError,json.JSONDecodeError)as error:
        raise ValueError('Native key origin intake: malformed JSON')from error


def _hash(value):return type(value)is str and SHA.fullmatch(value)is not None


def _bound(value):
    need(type(value)is dict and _hash(value.get('binding'))and
         value['binding']==digest({k:v for k,v in value.items()if k!='binding'}),'JSON binding')


def _terminal(value):
    need(type(value)is dict and value.get('terminal_returncode_observed')is True,
         'actual terminal observation required')
    row=value.get('terminal');need(type(row)is dict and type(row.get('exit_code'))is int and
         row['exit_code']==0 and type(row.get('output'))is str,'successful outer terminal')
    output=_json(row['output'].strip().encode())
    need(type(output)is dict and type(output.get('returncode'))is int and
         output['returncode']==0 and output.get('owner_verified')is True,'host-owned successful phase')


def verify(documents,*,expected_sha256):
    """Validate public evidence against pins supplied by the trusted launcher.

    expected_sha256 must come from the frozen host handoff, not from candidate
    JSON. The function intentionally gives no SDK or encrypted authority.
    """
    need(type(documents)is dict and set(documents)==DOCUMENTS and
         type(expected_sha256)is dict and set(expected_sha256)==DOCUMENTS,'exact evidence inventory')
    need(all(_hash(v)for v in expected_sha256.values()),'external raw identities')
    need(all(type(v)is bytes and 0<len(v)<=MAX_DOCUMENT for v in documents.values())and
         sum(map(len,documents.values()))<=MAX_TOTAL,'public evidence byte limits')
    for name,raw in documents.items():
        need(hashlib.sha256(raw).hexdigest()==expected_sha256[name],'frozen identity: '+name)
    d={name:_json(raw)for name,raw in documents.items()}
    for name in ('approval','library_plan','key_plan','public_origin'):_bound(d[name])
    a,l,k,lr,kr,o,m=(d[name]for name in ('approval','library_plan','key_plan',
                                      'library_report','key_report','public_origin','public_manifest'))
    need(a.get('status')=='approved'and a.get('authorization_source')=='direct_human_message'and
         a.get('library_fix_approved')is True and a.get('new_secret_key_epochs_approved')==1 and
         a.get('isolated_candidate_only')is True,'separate direct human scope')
    for plan in (l,k):
        need(plan.get('approved_proposal_binding')==a.get('approved_proposal_binding')and
             plan.get('special_approval_binding')==a['binding'],'approved plan scope')
    for report,plan in ((lr,l),(kr,k)):
        need(report.get('separate_approval_binding')==a['binding']and
             report.get('source_plan_binding')==plan['binding'],'actual report/plan/approval binding')
    _terminal(d['library_terminal']);_terminal(d['key_terminal'])
    need(lr.get('status')=='succeeded'and lr.get('project_CXX_units')==1 and
         lr.get('project_links')==1 and lr.get('new_keys')==0 and
         lr.get('native_evaluators')==0 and
         lr.get('original_library_source_and_keys_unchanged')is True and
         lr.get('exported_abi_symbol_types_names_unchanged')is True and
         lr.get('soname_and_required_libraries_unchanged')is True,'isolated successful library build')
    library=lr.get('candidate_library_sha256')
    need(_hash(library)and library!=lr.get('original_library_sha256')and
         library==k.get('candidate_library_sha256')==kr.get('candidate_library_sha256')==
         o.get('candidate_library_sha256'),'actual distinct candidate library')
    need(kr.get('status')=='new_epoch_created_and_sanity_checked_numeric_metrics_recorded'and
         kr.get('shared_batch_status')=='succeeded'and kr.get('new_secret_key_epochs')==1 and
         kr.get('new_public_keysets')==kr.get('new_relin_keysets')==1 and
         kr.get('new_Galois_keys')==25 and kr.get('new_encrypted_inputs')==4 and
         kr.get('Native_Boots')==kr.get('native_evaluators')==kr.get('API')==
         kr.get('downloads')==kr.get('retries')==kr.get('host_private_content_reads')==0,
         'exact one-epoch client execution scope')
    need(kr.get('private_key_security_sanity_passed')is True and
         kr.get('SDK_selected_or_promoted')is False and kr.get('numeric_policy')=='metrics-only',
         'key security and metrics-only scope')
    client=kr.get('client_report');need(type(client)is dict,'actual client result')
    sanity=client.get('dense_default_key_origin_check')
    need(type(sanity)is dict and sanity==o.get('private_key_origin_check')and
         sanity.get('status')=='passed'and all(sanity.get(name)is True for name in
         ('new_private_epoch','nonzero','both_signs_present','ternary_in_all_existing_QP','broad_dense_ternary_sanity'))and
         sanity.get('private_coefficients_counts_vectors_or_hashes_exported')is False and
         sanity.get('SDK_or_Boot_certified')is False,'client-only valid ternary origin check')
    epoch=o.get('epoch');origin=o.get('origin')
    need(type(epoch)is str and EPOCH.fullmatch(epoch)is not None and _hash(origin)and
         epoch==k.get('producer_epoch')==client.get('epoch')==m.get('epoch')and
         origin==k.get('origin_binding')==k.get('kernel_binding')==m.get('origin'),
         'actual epoch and public origin')
    need(o.get('format')=='poseidon-fresh-dense-key-public-client-origin-v1'and
         o.get('separate_approval_binding')==a['binding']and
         o.get('source_plan_binding')==k['binding']and
         o.get('not_SDK_or_whole_model_certification')is True and
         o.get('private_content_or_hash_not_read_or_exported_by_host')is True and
         o['binding']==kr.get('trusted_public_origin_binding'),'public origin scope')
    need(hashlib.sha256(documents['public_manifest']).hexdigest()==o.get('public_manifest_sha256')==
         kr.get('public_manifest_sha256'),'public manifest raw bytes')
    context=k.get('context');need(type(context)is dict,'frozen physical context')
    need(m.get('format')=='poseidon-bootstrap-public-piece-handoff-v1'and m.get('stage')=='input'and
         m.get('degree')==context.get('degree')and m.get('q')==context.get('q')and
         m.get('p')==context.get('p'),'unchanged original physical context')
    rows=m.get('files');need(type(rows)is list and rows==o.get('public_files')and len(rows)==213,
         'exact public key/ciphertext inventory')
    need(len({r.get('name')for r in rows})==len(rows)and
         all(type(r.get('name'))is str and '/'not in r['name']and '\\'not in r['name']and
             _hash(r.get('sha256'))and r.get('degree')==context['degree']and r.get('polys')==2
             for r in rows),'unchanged two-component public descriptors')
    inputs=sorted((r for r in rows if r.get('type')=='input'),key=lambda r:r['group'])
    need(len(inputs)==4 and [r['group']for r in inputs]==list(range(4)), 'four original input ciphertexts')
    phases=kr.get('native_phases');need(type(phases)is list and len(phases)==1 and
         phases[0].get('role')=='client-generate'and phases[0].get('returncode')==0 and
         phases[0].get('executable_sha256')==o.get('client_executable_sha256')and
         _hash(o.get('client_executable_sha256')),'actual client binary')
    metrics=client.get('checks');need(type(metrics)is list and len(metrics)==4,'four recorded numeric measurements')
    for index,row in enumerate(metrics):
        need(row.get('group')==index and row.get('operation')=='consumed'and
             row.get('compared_values')==context['degree']//2 and
             _hash(row.get('actual_sha256'))and _hash(row.get('reference_sha256')),'original numeric relation')
        for name in ('max_absolute_error','max_imaginary'):
            need(type(row.get(name))in(int,float)and math.isfinite(row[name])and row[name]>=0,
                 'finite recorded numeric values')
    # Historical passed/failed_values fields are intentionally not admission.
    result=dict(format='poseidon-verified-native-key-origin-v1',epoch=epoch,public_origin=origin,
        candidate_library_sha256=library,client_executable_sha256=o['client_executable_sha256'],
        approval_binding=a['binding'],library_plan_binding=l['binding'],key_plan_binding=k['binding'],
        trusted_public_origin_binding=o['binding'],public_manifest_sha256=o['public_manifest_sha256'],
        physical_context=copy.deepcopy(context),parameter_ids=copy.deepcopy(k['parameter_ids']),
        key_steps=copy.deepcopy(m['key_steps']),input_descriptors=copy.deepcopy(inputs),
        client_security_sanity_verified=True,private_material_or_hash_accessed=False,
        numeric_policy='metrics-only',thresholds=None,
        numeric_measurements=[{name:row[name]for name in ('group','operation','compared_values',
            'max_absolute_error','max_imaginary','actual_sha256','reference_sha256')}for row in metrics],
        evidence_sha256=copy.deepcopy(expected_sha256),public_file_bytes_attested=False,
        SDK_certified=False,Boot_certified=False,full_model_certified=False,execution_authority=False)
    result['binding']=digest(result)
    return result
