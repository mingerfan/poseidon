"""Bind retained genuine Conv children to source-derived ABI and public cover.

This staged data reader never authorizes a Runtime or a new Agent generation.
The caller supplies pinned original identity/receipt/ABI verification functions.
Original full package/source/current-execution checks remain mandatory upstream.
"""
import hashlib
import json
import math
import re
from pathlib import PurePosixPath

LIMIT=16*1024**2
STAGES=('generation','compilation','execution','validation')
RECEIPT='artifacts/generation-provenance.json'
IDENTITIES='artifacts/application-identities.json'


def need(ok,why):
    if not ok:raise ValueError(why)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(raw):return hashlib.sha256(raw).hexdigest()


def digest(value):return sha(canonical(value))


def hash_text(value):
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None,'SHA256 identity')
    return value


def loads(raw):
    need(type(raw) is bytes and len(raw)<=LIMIT,'bounded JSON bytes')
    def pairs(items):
        result={}
        for key,value in items:
            need(key not in result,'duplicate JSON key');result[key]=value
        return result
    def number(text):
        value=float(text);need(math.isfinite(value),'finite JSON number');return value
    return json.loads(raw,object_pairs_hook=pairs,parse_float=number,
        parse_constant=lambda _:need(False,'finite JSON number'))


def bound(value):
    need(type(value) is dict and value.get('binding')==digest({k:v for k,v in value.items()if k!='binding'}),
         'document binding')
    return value


def seal(value):
    need(len(canonical(value))<=131072,'bounded intake descriptor')
    return dict(value,binding=digest(value))


def relative(name):
    need(type(name) is str and 1<=len(name)<=255 and '\\'not in name,'relative package path')
    p=PurePosixPath(name)
    need(bool(p.parts) and not p.is_absolute() and str(p)==name and all(x not in ('..','.')for x in p.parts),
         'relative package path')
    return name


def inspect_child(index,read_relative,*,expected_package_binding,expected_model_sha256,
                  verify_identities,verify_generation,derive_abi):
    """Read original bytes, not preasserted role or qualification summaries.

    read_relative enforces regular/nonsymlink bounded filesystem reads. Its byte
    callback permits in-memory adversarial tests without copying old packages.
    The three verifier callbacks must come from the separately pinned sources.
    """
    need(type(index) is int and 0<=index<9,'component index')
    hash_text(expected_package_binding);hash_text(expected_model_sha256)
    raw_manifest=read_relative('manifest.json');m=bound(loads(raw_manifest))
    need(set(m)=={'format','files','compiler_configuration','origin','agent_generated',
        'binding','dependencies','model_provenance','validation'} and
        m['format']=='poseidon-dsl-bundle-v6','direct public v6 child package')
    need(m['binding']==expected_package_binding,'frozen original package binding')
    validation=m['validation']
    need(m['agent_generated']is True and type(validation) is dict and
        set(validation)=={'level','encrypted_execution','numerically_validated'} and
        validation['level']=='numerical' and validation['encrypted_execution']is True and
        validation['numerically_validated']is True,'genuine numerical child')
    need(m['model_provenance']=='not_supplied','direct public models only; no unaudited provenance')
    files=m['files'];need(type(files) is dict and 1<=len(files)<=1024,'package file count')
    hashes={}
    for name,expected in files.items():
        relative(name);hash_text(expected)
        raw=read_relative(name)
        need(type(raw) is bytes and len(raw)<=LIMIT and sha(raw)==expected,'original package file bytes')
        hashes[name]=expected
    def read(name):
        need(name in files,'required original artifact')
        raw=read_relative(name)
        need(type(raw) is bytes and len(raw)<=LIMIT and sha(raw)==files[name],'original artifact changed')
        return raw
    request_raw=read('request.json');request=loads(request_raw)
    model=loads(read('model.json'))
    need(canonical(model)==canonical(request['model']) and digest(model)==expected_model_sha256,
         'component model index/cover identity')
    need(request['request_id']==m['origin']['request_id'] and
        canonical(request['compiler_configuration'])==canonical(m['compiler_configuration']),
        'request and package identity/configuration')
    identity=m['dependencies']['application_identities']
    need(type(identity) is dict and set(identity)=={'file','sha256'} and
        identity['file']==IDENTITIES and files.get(IDENTITIES)==identity['sha256'],
        'original application identities pointer')
    identities=verify_identities(loads(read(IDENTITIES)))
    stages={stage:hash_text(identities[stage]['sha256'])for stage in STAGES}
    pointer=m['origin'].get('generation_provenance')
    need(type(pointer) is dict and set(pointer)=={'file','sha256'} and
        pointer['file']==RECEIPT and files.get(RECEIPT)==pointer['sha256'],
        'mandatory genuine generation receipt')
    source_raw=read('candidate.py');source=source_raw.decode('utf-8')
    receipt=loads(read(RECEIPT))
    generation=verify_generation(receipt,request,source,
        generation_identity_sha256=stages['generation'])
    need(generation['agent_generated']is True and generation['generations']==1 and
        generation['http_attempts']==1 and generation['repairs_used']==0,
        'original single-generation zero-repair cohort')
    selected_raw=read('artifacts/candidate_trace.entry.mlir')
    entry=loads(read('artifacts/entry-selection.json'))
    need(entry['entry']=='_hecate_golden' and entry['function_bodies_unchanged']is True and
        entry['selected_sha256']==sha(selected_raw) and
        entry['original_sha256']==sha(read('artifacts/candidate_trace.compact.mlir')),
        'original selected entry receipt')
    lowered_raw=read('artifacts/lowered.ckks.mlir')
    raw_plan=read('artifacts/lowered._hecate_golden.runtime-plan.json')
    plan_raw=read('artifacts/bound/plan.json')
    mapping=derive_abi(request,source,selected_raw.decode(),lowered_raw.decode(),
        loads(raw_plan),loads(plan_raw))
    io=dict(inputs=[dict(value_id=row['external_value_id'],role={k:v for k,v in row.items()
        if k not in ('abi_index','external_value_id')})for row in mapping['inputs']],
        outputs=[dict(value_id=row['final_value_id'],role=dict(
            logical={k:v for k,v in row.items()if k not in ('final_index','final_value_id')},
            representation=request['layout']['output_representation'],
            input_slot_period=request['layout']['input_slot_period'],
            output_selectors_sha256=mapping['output_selectors_sha256']))for row in mapping['outputs']])
    spec_raw=read('artifacts/operator-spec.json')
    bundle_raw=read('artifacts/bound/bundle/manifest.json')
    # The descriptor binds historical stage/backend data; it does not compare
    # those identities to a current process, future49 policy, or new linker.
    return seal(dict(format='poseidon-retained-genuine-Conv-child-intake-v1',index=index,
        original_manifest_raw_sha256=sha(raw_manifest),program_binding=m['binding'],
        model_sha256=digest(model),request_id=request['request_id'],request_sha256=sha(request_raw),
        candidate_sha256=sha(source_raw),generation_receipt_binding=receipt['binding'],
        generation_receipt_sha256=files[RECEIPT],generation= generation,
        application_identities_sha256=files[IDENTITIES],application_manifest_sha256=identities['manifest_sha256'],
        original_stage_sha256=stages,backend_identity_sha256=digest(m['dependencies']['backend_identity']),
        compiler_configuration_sha256=digest(m['compiler_configuration']),
        selected_IR_sha256=sha(selected_raw),lowered_IR_sha256=sha(lowered_raw),
        original_plan_sha256=sha(raw_plan),bound_plan_sha256=sha(plan_raw),
        operator_spec_sha256=sha(spec_raw),bundle_manifest_sha256=sha(bundle_raw),
        io=io,io_sha256=digest(io),package_files_verified=len(hashes),
        original_check_bundle_invoked=False,current_execution_dependencies_checked=False,
        execution_authority=False,numerical_sum_acceptance=False))


def pending_cover_intake(cover,children,*,require_complete=False):
    """An explicit missing-set gate, never a new executable program package."""
    bound(cover)
    need(cover.get('format')=='poseidon-exact-public-Conv-kernel-cover-v1' and
        cover.get('algebraic_cover')is True and cover.get('execution_authority')is False,
        'original data-only coefficient cover')
    rows=cover.get('pieces')
    need(type(rows) is list and 2<=len(rows)<=9,'source cover count')
    for i,row in enumerate(rows):
        need(type(row.get('index')) is int and row['index']==i,'source cover order')
        hash_text(row['model_sha256'])
    need(type(children) is list and len(children)<=len(rows),'genuine child count')
    indices=[];packages=set();histories={}
    for child in children:
        bound(child)
        need(child.get('format')=='poseidon-retained-genuine-Conv-child-intake-v1' and
            child.get('execution_authority')is False,'retained child descriptor kind')
        i=child['index'];need(type(i) is int and 0<=i<len(rows) and i not in indices,'unique in-range child index')
        need(child['model_sha256']==rows[i]['model_sha256'],'genuine child belongs to exact cover index')
        p=hash_text(child['program_binding']);need(p not in packages,'unique original program package');packages.add(p)
        indices.append(i)
        stages=child['original_stage_sha256'];need(set(stages)==set(STAGES),'four historical stages')
        for key in STAGES:hash_text(stages[key])
        key=digest(dict(stages=stages,backend=child['backend_identity_sha256'],
            configuration=child['compiler_configuration_sha256']))
        histories.setdefault(key,[]).append(i)
    need(indices==sorted(indices),'stable source child order')
    missing=[i for i in range(len(rows))if i not in indices]
    need(type(require_complete)is bool,'complete-set gate flag')
    if require_complete:
        need(not missing,'missing genuine components: '+str(missing))
        need(len(histories)==1,'mixed historical identities require explicit source compatibility admission')
    return seal(dict(format='poseidon-pending-Conv-composition-source-intake-v1',
        coefficient_cover_binding=cover['binding'],source_model_raw_sha256=cover['source_model_raw_sha256'],
        source_node_id=cover['source_node_id'],source_weight=cover['source_weight'],
        required_indices=list(range(len(rows))),qualified_indices=indices,missing_indices=missing,
        children=[dict(index=r['index'],intake_binding=r['binding'],program_binding=r['program_binding'],
            model_sha256=r['model_sha256'],bound_plan_sha256=r['bound_plan_sha256'],io_sha256=r['io_sha256'])for r in children],
        historical_identity_groups=histories,complete_retained_child_set=not missing,
        status='requires_original_package_source_CPP_and_numerical_admission',
        trusts_pinned_reader_callbacks=True,current_execution_compatibility=False,
        execution_authority=False,numerical_sum_acceptance=False,complete_source_graph_accepted=False))
