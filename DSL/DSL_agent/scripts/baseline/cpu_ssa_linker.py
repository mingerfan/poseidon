"""Pure RuntimePlan-v1 sum linker, staged outside production.

The caller must supply qualified, source-derived I/O roles. This transformation
does not qualify children, grant execution authority, or prove numerical sums.
It reads blob bytes through supplied callbacks and performs no filesystem writes.
"""
import copy
import hashlib
import json
import math
import re
import struct

MAX_OPS = 4096
MAX_VALUES = 8192
MAX_FILE = 16*1024**2
MAX_DESCRIPTOR = 128*1024
MAX_SLOTS = 16384
HOST = dict(kind='host', rank=0)
CONTEXT = 'poseidon-agent-n15-q13-p1-tc128-v1'
ATTRS = dict(rotate={'steps'}, mod_switch={'target_level'},
             rescale={'target_level', 'target_scale_log2'})
BINARY = {'add_cc', 'add_cp', 'sub_cc', 'sub_cp', 'mul_cc', 'mul_cp'}
UNARY = {'negate', 'rotate', 'rescale', 'mod_switch', 'relinearize'}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return 'sha256:'+hashlib.sha256(raw).hexdigest()


def exact(value, keys, why):
    require(type(value) is dict and set(value) == set(keys), why)


def integer(value, lower, upper, why):
    require(type(value) is int and lower <= value <= upper, why)


def value_id(value):
    require(type(value) is str and re.fullmatch(r'0|[1-9][0-9]*', value) is not None and
            len(value) <= 20 and int(value) <= 2**64-1, 'canonical uint64 ValueId')
    return value


def content_name(value):
    require(type(value) is str and re.fullmatch('sha256:[0-9a-f]{64}', value) is not None,
            'canonical SHA256 content')


def loads(raw):
    require(type(raw) is bytes and len(raw) <= MAX_FILE, 'bounded JSON bytes')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, 'nonfinite JSON'))


def physical(value):
    return {key: item for key, item in value.items() if key != 'id'}


def fixed_spec(raw):
    spec = loads(raw)
    integer(spec['spec_format_version'],2,2,'OperatorSpec version2')
    integer(spec['version'],1,1,'OperatorSpec revision1')
    require(spec['spec_id'] == 'poseidon-agent-cpu-v1' and
            spec['target_id'] == 'poseidon-ckks-cpu', 'fixed CPU OperatorSpec')
    require(canonical(spec['context']) == canonical(dict(context_id=CONTEXT, poly_degree=32768,
                rns_moduli_log2=[60]*13, max_modulus_log2=60, default_scale_log2=45)) and
            canonical(spec['levels']) == canonical(dict(lower_bound=0, upper_bound=12)) and
            spec['operators']['boot']['supported'] is False and spec['boot_profiles'] == [],
            'unchanged CPU/security/bootstrap profile')
    return spec


def finite_vector(values):
    require(type(values) is list and 1 <= len(values) <= MAX_SLOTS, 'bounded plaintext slots')
    try:
        valid = all(type(value) in (int, float) and math.isfinite(value) for value in values)
    except OverflowError:
        valid = False
    require(valid, 'finite plaintext f64 values')


def verify_plan(plan, spec_raw, rotations):
    """Strict v1 CPU structural/metadata checks; original backend verification still required."""
    spec = fixed_spec(spec_raw)
    keys = {'format_version','plan_id','target','values','external_inputs',
            'initialization','execution','finalization','final_outputs'}
    require(type(plan) is dict and set(plan) in (keys, keys|{'plaintext_bundle'}), 'v1 plan fields')
    integer(plan['format_version'], 1, 1, 'RuntimePlan version1 only')
    value_id(plan['plan_id'])
    target = dict(target_id='poseidon-ckks-cpu', capability_version=1,
        operator_spec=dict(id=spec['spec_id'], version=1, source_sha256=digest(spec_raw)),
        world_size=1, device_counts=[0])
    require(plan['target'] == target and canonical(plan['target']) == canonical(target),
            'exact CPU target/spec raw SHA')
    require(type(plan['values']) is list and 1 <= len(plan['values']) <= MAX_VALUES, 'value bound')
    require(all(type(plan[phase]) is list for phase in ('initialization','execution','finalization')) and
            plan['finalization'] == [], 'v1 phases/finalization')
    require(len(plan['initialization'])+len(plan['execution']) <= MAX_OPS, 'instruction bound')
    require(type(rotations) is list and len(rotations) <= MAX_OPS, 'rotation authority bound')
    for step in rotations:
        integer(step, -2**31, 2**31-1, 'rotation authority integer')
        require(step % MAX_SLOTS != 0, 'nontrivial authorized rotation')
    authorized = set(rotations)
    values = {}
    for value in plan['values']:
        exact(value, {'id','kind','place','context','level','scale_log2','ntt','components'},
              'ValueDesc exact fields')
        key = value_id(value['id'])
        require(key not in values, 'duplicate ValueDesc')
        require(value['kind'] in ('plaintext','ciphertext') and value['context'] == CONTEXT and
                value['place'] == HOST and canonical(value['place']) == canonical(HOST) and
                value['ntt'] is True, 'fixed host/context/NTT value')
        integer(value['level'], 0, 12, 'value level')
        integer(value['scale_log2'], 0, 60*(value['level']+1)-1, 'value scale/modulus capacity')
        integer(value['components'], 1 if value['kind']=='plaintext' else 2,
                1 if value['kind']=='plaintext' else 3, 'value components')
        values[key] = value
    require(type(plan['external_inputs']) is list and 1 <= len(plan['external_inputs']) <= 17,
            'external input bound')
    defined = set()
    mentioned = set()
    for key in plan['external_inputs']:
        value_id(key)
        require(key in values and key not in defined, 'unique declared external')
        value = values[key]
        require((value['kind'],value['level'],value['scale_log2'],value['components']) ==
                ('ciphertext',12,45,2), 'fixed external physical ABI')
        defined.add(key)
        mentioned.add(key)
    contents = set()
    encodes = []
    ordinal = 0
    for phase in ('initialization','execution'):
        for op in plan[phase]:
            require(type(op) is dict and op.get('kind') in ('encode','compute'), 'supported v1 instruction')
            integer(op.get('ordinal'), ordinal, ordinal, 'contiguous original ordinal')
            ordinal += 1
            key = value_id(op.get('output'))
            require(key in values and key not in defined, 'SSA destination declared and fresh')
            result = values[key]
            if op['kind'] == 'encode':
                exact(op, {'ordinal','kind','output','payload'}, 'encode fields')
                require(phase == 'initialization' and result['kind']=='plaintext', 'encode phase/type')
                payload = op['payload']
                require(type(payload) is dict, 'encode payload object')
                if payload.get('kind') == 'inline':
                    exact(payload, {'kind','values'}, 'inline payload fields')
                    finite_vector(payload['values'])
                else:
                    exact(payload, {'kind','content'}, 'bundle payload fields')
                    require(payload['kind'] == 'bundle', 'encode payload kind')
                    content_name(payload['content'])
                    contents.add(payload['content'])
                encodes.append((op, result))
            else:
                kind = op.get('op')
                require(kind in BINARY|UNARY and spec['operators'][kind]['supported'] is True,
                        'supported CPU compute capability')
                opkeys = {'ordinal','kind','op','place','inputs','output'}
                exact(op, opkeys|({'attrs'} if kind in ATTRS else set()), 'compute fields/attrs')
                require(op['place'] == HOST and canonical(op['place']) == canonical(HOST), 'compute Host0')
                require(type(op['inputs']) is list and len(op['inputs']) == (2 if kind in BINARY else 1),
                        'compute arity')
                for source in op['inputs']:
                    value_id(source)
                    require(source in defined, 'use before definition')
                    mentioned.add(source)
                a = values[op['inputs'][0]]
                require(a['kind']=='ciphertext', 'first operand ciphertext')
                expected = physical(a)
                if kind in ATTRS:
                    exact(op['attrs'], ATTRS[kind], 'compute attribute fields')
                if kind in BINARY:
                    b = values[op['inputs'][1]]
                    require(b['kind'] == ('ciphertext' if kind.endswith('cc') else 'plaintext') and
                            all(a[k]==b[k] for k in ('place','context','level','ntt')), 'binary operand metadata')
                    if kind.startswith('mul'):
                        expected['scale_log2'] += b['scale_log2']
                        if kind.endswith('cc'):
                            expected['components'] = a['components']+b['components']-1
                    else:
                        require(a['scale_log2']==b['scale_log2'] and
                                (kind.endswith('cp') or a['components']==b['components']),
                                'addition/subtraction metadata')
                elif kind == 'rotate':
                    step = op['attrs']['steps']
                    integer(step, -2**31, 2**31-1, 'rotation step integer')
                    require(step % MAX_SLOTS != 0 and step in authorized and a['components']==2,
                            'rotation authority/metadata')
                elif kind == 'mod_switch':
                    level = op['attrs']['target_level']
                    integer(level, 0, a['level']-1, 'strict mod-switch level drop')
                    expected['level'] = level
                elif kind == 'rescale':
                    attrs = op['attrs']
                    integer(attrs['target_level'], a['level']-1, a['level']-1, 'single rescale level')
                    integer(attrs['target_scale_log2'], a['scale_log2']-60, a['scale_log2']-60,
                            'rescale may not override interpretation')
                    require(attrs['target_level'] >= 0 and attrs['target_scale_log2'] >= 0 and
                            spec['operators']['rescale']['max_levels_per_op'] == 1, 'rescale capacity')
                    expected.update(level=attrs['target_level'], scale_log2=attrs['target_scale_log2'])
                elif kind == 'relinearize':
                    require(a['components']==3, 'relinearize components')
                    expected['components'] = 2
                require(physical(result) == expected, 'compute output metadata propagation')
            defined.add(key)
            mentioned.add(key)
    require(type(plan['final_outputs']) is list and 1 <= len(plan['final_outputs']) <= 16 and
            len(set(plan['final_outputs'])) == len(plan['final_outputs']), 'distinct final output bound')
    for key in plan['final_outputs']:
        value_id(key)
        require(key in defined and values[key]['kind']=='ciphertext', 'defined ciphertext final')
        mentioned.add(key)
    require(mentioned == set(values), 'unmentioned ValueDesc')
    require(('plaintext_bundle' in plan) == bool(contents), 'bundle iff bundle encode')
    if contents:
        exact(plan['plaintext_bundle'], {'id','version','manifest_sha256'}, 'bundle reference fields')
        ref = plan['plaintext_bundle']
        require(type(ref['id']) is str and 1 <= len(ref['id']) <= 128, 'bundle id')
        integer(ref['version'], 1, 2**31-1, 'bundle version')
        content_name(ref['manifest_sha256'])
    require(len(canonical(plan)) <= MAX_FILE, 'plan file bound')
    return values, contents, encodes


def verify_bundle(plan, raw, reader, contents, encodes):
    """Validate exact original bytes; return data identities, never float-reencode."""
    if not contents:
        require(raw is None, 'no bundle manifest without bundle encode')
        blobs = {}
    else:
        require(type(raw) is bytes and len(raw) <= MAX_DESCRIPTOR and
                digest(raw) == plan['plaintext_bundle']['manifest_sha256'], 'bundle manifest raw SHA')
        manifest = loads(raw)
        exact(manifest, {'bundle_format_version','bundle_id','version','blobs'}, 'bundle manifest fields')
        integer(manifest['bundle_format_version'], 1, 1, 'bundle format version')
        require(manifest['bundle_id'] == plan['plaintext_bundle']['id'] and
                type(manifest['version']) is int and manifest['version'] == plan['plaintext_bundle']['version'],
                'bundle reference identity')
        require(type(manifest['blobs']) is list and len(manifest['blobs']) <= MAX_OPS, 'bundle entry bound')
        blobs = {}
        for entry in manifest['blobs']:
            exact(entry, {'content','byte_length'}, 'bundle entry fields')
            name = entry['content']
            content_name(name)
            require(name not in blobs, 'duplicate bundle content')
            integer(entry['byte_length'], 8, 8*MAX_SLOTS, 'bounded f64 blob')
            require(entry['byte_length'] % 8 == 0, 'f64 blob alignment')
            data = reader(name)
            require(type(data) is bytes and len(data)==entry['byte_length'] and digest(data)==name,
                    'original blob bytes/SHA/length')
            finite_vector([value for (value,) in struct.iter_unpack('<d',data)])
            blobs[name] = data
        require(set(blobs) == contents, 'exact referenced bundle contents')
    for op, desc in encodes:
        payload = op['payload']
        slots = payload['values'] if payload['kind']=='inline' else [
            value for (value,) in struct.iter_unpack('<d',blobs[payload['content']])]
        magnitude = max(abs(value) for value in slots)
        require(magnitude==0 or math.ceil(math.log2(magnitude))+desc['scale_log2']+1 <
                60*(desc['level']+1), 'plaintext encoding capacity')
    return blobs


def role_json(role):
    """Canonical JSON must not silently coerce Python keys or custom objects."""
    pending=[(role,0)]
    visited=0
    while pending:
        value,depth=pending.pop()
        visited+=1
        require(visited<=16384 and depth<=32,'semantic role JSON work bound')
        if type(value) is dict:
            require(all(type(key) is str for key in value),'semantic role JSON string keys')
            pending.extend((item,depth+1) for item in value.values())
        elif type(value) is list:
            pending.extend((item,depth+1) for item in value)
        else:
            require(type(value) in (str,int,float,bool,type(None)),'semantic role JSON builtin value')
            if type(value) is float:
                require(math.isfinite(value),'semantic role JSON finite number')


def io_keys(io, plan):
    exact(io, {'inputs','outputs'}, 'I/O role evidence fields')
    result = {}
    for field, ids in [('inputs',plan['external_inputs']),('outputs',plan['final_outputs'])]:
        require(type(io[field]) is list and len(io[field])==len(ids), 'I/O role count')
        keys = []
        for index, row in enumerate(io[field]):
            exact(row, {'value_id','role'}, 'I/O role mapping fields')
            require(row['value_id']==ids[index] and type(row['role']) is dict and bool(row['role']),
                    'ABI order/value binding')
            role_json(row['role'])
            key = canonical(row['role'])
            require(len(key) <= MAX_DESCRIPTOR and key not in keys, 'distinct bounded semantic roles')
            keys.append(key)
        result[field] = keys
    return result


def link_sum(children, *, plan_id, bundle_id):
    """Link corresponding outputs by stable left-fold add_cc, with proven input fanout.

    Each child has id, plan_raw, spec_raw, bundle_raw (or None), read_blob,
    rotations and io. io is provided by the separate qualified-source ABI loader.
    The result is structural data plus provenance maps, not an executable package.
    """
    require(type(children) is list and 2 <= len(children) <= 9, 'composition child count2..9')
    value_id(plan_id)
    require(type(bundle_id) is str and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',bundle_id) is not None,
            'new bounded bundle id')
    checked = []
    names = set()
    first_spec = None
    first_target = None
    first_roles = None
    union = {}
    sources = {}
    rotations = set()
    for index, child in enumerate(children):
        exact(child, {'id','plan_raw','spec_raw','bundle_raw','read_blob','rotations','io'}, 'child fields')
        name = child['id']
        require(type(name) is str and re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',name) is not None and
                name not in names, 'unique child identity')
        names.add(name)
        plan = loads(child['plan_raw'])
        values, contents, encodes = verify_plan(plan,child['spec_raw'],child['rotations'])
        blobs = verify_bundle(plan,child['bundle_raw'],child['read_blob'],contents,encodes)
        roles = io_keys(child['io'],plan)
        if index == 0:
            first_spec, first_target, first_roles = child['spec_raw'], plan['target'], roles
        else:
            require(child['spec_raw']==first_spec and plan['target']==first_target, 'same target/spec bytes')
            require(roles==first_roles, 'same ordered source-derived semantic roles')
            prior_values = checked[0]['values']
            prior_plan = checked[0]['plan']
            for field in ('external_inputs','final_outputs'):
                require(all(physical(values[b])==physical(prior_values[a])
                            for a,b in zip(prior_plan[field],plan[field])), 'corresponding physical I/O metadata')
        for content, raw in blobs.items():
            if content in union:
                require(union[content]==raw, 'content-addressed dedup requires exact bytes')
            else:
                union[content]=raw
                sources[content]=dict(child=index,byte_length=len(raw))
        rotations.update(child['rotations'])
        checked.append(dict(plan=plan,values=values))

    first = checked[0]
    e = len(first['plan']['external_inputs'])
    k = len(first['plan']['final_outputs'])
    expected_ops = sum(len(c['plan']['initialization'])+len(c['plan']['execution']) for c in checked)+(len(children)-1)*k
    expected_values = sum(len(c['values']) for c in checked)-(len(children)-1)*e+(len(children)-1)*k
    require(expected_ops <= MAX_OPS and expected_values <= MAX_VALUES, 'aggregate original resource bounds')
    linked = dict(format_version=1,plan_id=plan_id,target=copy.deepcopy(first_target),values=[],
                  external_inputs=[str(i) for i in range(e)],initialization=[],execution=[],
                  finalization=[],final_outputs=[])
    for index,key in enumerate(first['plan']['external_inputs']):
        linked['values'].append(dict(copy.deepcopy(first['values'][key]),id=str(index)))
    maps = []
    next_id = e
    for child in checked:
        mapping = dict(zip(child['plan']['external_inputs'],linked['external_inputs']))
        for value in child['plan']['values']:
            if value['id'] in mapping:
                continue
            new_id = str(next_id)
            next_id += 1
            mapping[value['id']] = new_id
            linked['values'].append(dict(copy.deepcopy(value),id=new_id))
        require(len(set(mapping.values()))==len(mapping), 'injective child map')
        maps.append(mapping)
    ordinal = 0
    for phase in ('initialization','execution'):
        for child,mapping in zip(checked,maps):
            for old in child['plan'][phase]:
                op = copy.deepcopy(old)
                op['ordinal'] = ordinal
                ordinal += 1
                op['output'] = mapping[old['output']]
                if op['kind']=='compute':
                    op['inputs'] = [mapping[key] for key in old['inputs']]
                linked[phase].append(op)
    for output_index in range(k):
        current = maps[0][first['plan']['final_outputs'][output_index]]
        for child,mapping in zip(checked[1:],maps[1:]):
            right = mapping[child['plan']['final_outputs'][output_index]]
            fresh = str(next_id)
            next_id += 1
            metadata = physical(first['values'][first['plan']['final_outputs'][output_index]])
            linked['values'].append(dict(copy.deepcopy(metadata),id=fresh))
            linked['execution'].append(dict(ordinal=ordinal,kind='compute',op='add_cc',
                place=dict(HOST),inputs=[current,right],output=fresh))
            ordinal += 1
            current = fresh
        linked['final_outputs'].append(current)
    bundle_raw = None
    if union:
        bundle = dict(bundle_format_version=1,bundle_id=bundle_id,version=1,
            blobs=[dict(content=name,byte_length=len(union[name])) for name in sorted(union)])
        bundle_raw = canonical(bundle)
        require(len(bundle_raw) <= MAX_DESCRIPTOR, 'merged bundle manifest bound')
        linked['plaintext_bundle'] = dict(id=bundle_id,version=1,manifest_sha256=digest(bundle_raw))
    require(ordinal==expected_ops and len(linked['values'])==expected_values, 'aggregate count equations')
    values, contents, encodes = verify_plan(linked,first_spec,sorted(rotations))
    verify_bundle(linked,bundle_raw,lambda name:union[name],contents,encodes)
    proof = dict(format='poseidon-data-only-SSA-sum-map-v1',
        children=[dict(id=child['id'],plan_sha256=digest(child['plan_raw']),
            bundle_manifest_sha256=None if child['bundle_raw'] is None else digest(child['bundle_raw']),
            io_sha256=digest(canonical(child['io'])),value_map=mapping)
            for child,mapping in zip(children,maps)],
        plan_sha256=digest(canonical(linked)),spec_sha256=digest(first_spec),
        bundle_manifest_sha256=None if bundle_raw is None else digest(bundle_raw),
        shared_external_roles=[dict(value_id=key,role=copy.deepcopy(row['role']))
            for key,row in zip(linked['external_inputs'],children[0]['io']['inputs'])],
        final_roles=[dict(value_id=key,role=copy.deepcopy(row['role']))
            for key,row in zip(linked['final_outputs'],children[0]['io']['outputs'])],
        addition_order='left fold in explicit child order for each corresponding final',
        instruction_count=ordinal,value_count=len(values),bundle_blobs=len(union),
        execution_authority=False,numerical_sum_acceptance=False)
    require(len(canonical(proof)) <= MAX_DESCRIPTOR, 'parent descriptor bound')
    return dict(plan=linked,bundle_raw=bundle_raw,bundle_sources=sources,proof=proof)
