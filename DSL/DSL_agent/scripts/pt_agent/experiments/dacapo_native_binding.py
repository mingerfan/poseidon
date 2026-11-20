"""Explicit N17 binding and key-free inventory for the original DaCapo plan.

Only the own-client adapter is described here. Original HEAAN concrete primes,
security and full CKKS accuracy remain unverified. No precision mapping applies
to the original plan. Scalar broadcasting is a stated interpretation of the
original DSL, and full execution requires independent per-site provenance and
three saved-training-probe mathematical evidence.
"""

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import re
import struct


CONTEXT = 'dacapo-original-n17-r51-export'
BOOT_PROFILE = 'dacapo-original-l3-to16-s51-export'
VARIANT = 'dacapo-own-lattigo-n17-q55-51x16-p61x3-scale51-slots16-lazy1'
OPS = dict(mod_switch=1, rotate=1, mul_cp=2, rescale=1, add_cc=2,
           add_cp=2, boot=1, mul_cc=2, relinearize=1, negate=1)
SOURCE_FACTORY = 'pt_agent.experiments.r13_common_models:resnet20_silu'


class UnsupportedPlan(ValueError):
    pass


def native_parameters():
    return dict(variant=VARIANT, log_n=17, logical_log_slots=16, slots=65536,
                log_q=[55] + [51] * 16, log_p=[61] * 3, max_level=16,
                log_default_scale=51, bootstrap_log_n=17, bootstrap_log_slots=16,
                bootstrap_log_p=[61] * 4, bootstrap_cache_capacity=1,
                residual_cache_capacity=1, plaintext_cache_capacity=1,
                source='Existing R27 original Orbit constructor literal, now an explicitly named DaCapo own-client binding',
                original_heaan_prime_equivalence=None, actual_q=None, actual_p=None,
                security_acceptance=None, original_plan_modified=False,
                precision_mapping_applied=False, bootstrap_default_algorithm_preserved=True)


def content_filename(content):
    """Treat the existing string as an opaque locator, without digest checks."""
    if not isinstance(content, str) or ':' not in content:
        raise UnsupportedPlan('Plaintext content locator needs a prefix and safe leaf')
    _, leaf = content.split(':', 1)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', leaf):
        raise UnsupportedPlan('Unsafe plaintext content locator')
    return leaf + '.bin'


def _finite_scalar(value):
    return type(value) in (int, float) and math.isfinite(value)


def _metadata_equation(op, operands, output):
    a = operands[0]
    level, scale, components = a['level'], a['scale_log2'], a['components']
    kind = op['op']
    attrs = op.get('attrs') or {}
    if len(operands) == 2:
        b = operands[1]
        if b['level'] != level or b['context'] != a['context']:
            raise UnsupportedPlan('Binary operand context or level differs')
        if b['kind'] != ('plaintext' if kind.endswith('_cp') else 'ciphertext'):
            raise UnsupportedPlan('Binary operand kind differs')
        if kind.startswith('add'):
            if b['scale_log2'] != scale:
                raise UnsupportedPlan('Original nominal addition scales differ')
            components = max(components, b['components'])
        else:
            scale += b['scale_log2']
            components += b['components'] - 1
    if kind in ('mod_switch', 'rescale'):
        target = attrs.get('target_level')
        if type(target) is not int or not 0 <= target < level:
            raise UnsupportedPlan('Invalid level drop')
        if kind == 'rescale':
            if target != level - 1 or attrs.get('target_scale_log2') != scale - 51:
                raise UnsupportedPlan('Original rescale must drop one51-bit nominal prime')
            scale -= 51
        level = target
    elif kind == 'boot':
        if (attrs != dict(implementation='native', operator_profile=BOOT_PROFILE,
                          target_level=16, target_scale_log2=51, target_components=2)
                or (level, scale, components) != (3, 51, 2)):
            raise UnsupportedPlan('Original native bootstrap contract must be3/51 ->16/51')
        level, scale, components = 16, 51, 2
    elif kind == 'relinearize':
        if components != 3:
            raise UnsupportedPlan('Relinearize requires three components')
        components = 2
    elif kind == 'rotate':
        if type(attrs.get('steps')) is not int or components != 2:
            raise UnsupportedPlan('Rotate requires integer steps and two components')
    if (output['level'], output['scale_log2'], output['components']) != (level, scale, components):
        raise UnsupportedPlan('Output metadata differs from original operation equation')


def _same_source_plan(source_plan, plan_path):
    """Bind proof metadata to one existing plan without content verification."""
    if not isinstance(source_plan, str) or not source_plan or plan_path is None:
        return False
    proof = Path(source_plan)
    if not proof.is_absolute():
        return False
    try:
        actual = Path(plan_path).resolve(strict=True)
        return actual.is_file() and proof.resolve(strict=True) == actual
    except (OSError, RuntimeError, ValueError, TypeError):
        return False


def _source_gates(plan, inline_ids, source_proof, source_math, *, public_fixture, plan_path=None):
    if public_fixture:
        if len(plan['execution']) > 12 or plan.get('target',{}).get('target_id') != 'dacapo-own-native-public-fixture':
            raise UnsupportedPlan('Public fixture broadcasting is limited to twelve operations')
        return dict(per_site_provenance_verified=True, original_training_math_verified=False,
                    scope='Explicit public synthetic fixture; no original-model acceptance',
                    execution_scope_ready=True, unsupported_reasons=[])
    reasons = []
    provenance = False
    if isinstance(source_proof, dict) and source_proof.get('status') == 'scalar_sites_verified':
        rows = source_proof.get('sites', [])
        provenance = (_same_source_plan(source_proof.get('source_plan'), plan_path)
            and source_proof.get('source_factory') == SOURCE_FACTORY
            and source_proof.get('original_dsl_scalar_broadcast_interpretation') is True
            and source_proof.get('original_plan_modified') is False
            and len(rows) == len(inline_ids) and {r.get('value_id') for r in rows} == set(inline_ids)
            and all(r.get('interpretation') == 'full-slot-scalar' and isinstance(r.get('source_path'), str)
                    and type(r.get('source_line')) is int and r['source_line'] > 0 for r in rows))
    if not provenance:
        reasons.append('RuntimePlan drops Python scalar versus one-element vector provenance; all inline sites need independent evidence for this source plan')
    mathematics = (isinstance(source_math, dict) and source_math.get('status') == 'strict_math_passed'
        and _same_source_plan(source_math.get('source_plan'), plan_path)
        and source_math.get('source_factory') == SOURCE_FACTORY
        and source_math.get('sample_ids') == ['train:0', 'train:1', 'train:2']
        and source_math.get('original_poly_preserved') is True
        and source_math.get('original_pack_bb32_preserved') is True
        and source_math.get('original_plan_modified') is False
        and source_math.get('absolute_tolerance') == 1e-12 and source_math.get('relative_tolerance') == 1e-12
        and source_math.get('all_three_passed') is True)
    if not mathematics:
        reasons.append('Original physical32 Poly and full RuntimePlan require three strict saved-training-probe comparisons for this source plan')
    return dict(per_site_provenance_verified=provenance,
                original_training_math_verified=mathematics, execution_scope_ready=provenance and mathematics,
                scope='Original DSL scalar broadcast interpretation; no claim about the old Poseidon vector-encoder path',
                unsupported_reasons=reasons)


def plan_inventory(plan, manifest, *, source_proof=None, source_math=None, public_fixture=False, plan_path=None):
    """Validate every symbolic operation, then count the complete last-use set."""
    if plan.get('format_version') != 1 or plan.get('finalization') != []:
        raise UnsupportedPlan('Only original format1 with empty finalization is supported')
    values = {}
    for v in plan.get('values', []):
        name = v.get('id')
        if (not isinstance(name, str) or name in values or v.get('context') != CONTEXT
                or v.get('place') != dict(kind='host', rank=0) or v.get('ntt') is not True
                or type(v.get('level')) is not int or not 0 <= v['level'] <= 16
                or type(v.get('scale_log2')) is not int or not 0 <= v['scale_log2'] <= 1023
                or v.get('kind') not in ('plaintext', 'ciphertext')
                or v.get('components') not in ((1,) if v['kind']=='plaintext' else (2,3))):
            raise UnsupportedPlan('Invalid original value metadata: ' + str(name))
        values[name] = v
    if len(plan.get('external_inputs', [])) != 1 or len(plan.get('final_outputs', [])) != 1:
        raise UnsupportedPlan('One input and one output required')
    for name in plan['external_inputs'] + plan['final_outputs']:
        if name not in values or values[name]['kind'] != 'ciphertext' or values[name]['components'] != 2:
            raise UnsupportedPlan('Ciphertext I/O metadata is missing')
    bundle = plan.get('plaintext_bundle') or {}
    if (manifest.get('bundle_format_version'), manifest.get('version'), manifest.get('bundle_id')) != (1, 1, bundle.get('id')) or bundle.get('version') != 1:
        raise UnsupportedPlan('Bundle identity or version differs')
    blobs = {}
    for blob in manifest.get('blobs', []):
        name = blob.get('content')
        content_filename(name)
        if name in blobs or blob.get('byte_length') != 65536 * 8:
            raise UnsupportedPlan('Original bundle raw vector must contain65536 float64 values')
        blobs[name] = blob
    defined, encodes, inline = set(plan['external_inputs']), {}, []
    for item in plan.get('initialization', []):
        name, payload = item.get('output'), item.get('payload') or {}
        if item.get('kind') != 'encode' or name in defined or name not in values or values[name]['kind'] != 'plaintext':
            raise UnsupportedPlan('Invalid plaintext initialization')
        if payload.get('kind') == 'bundle':
            if payload.get('content') not in blobs:
                raise UnsupportedPlan('Bundle reference has no metadata entry')
        elif payload.get('kind') == 'inline':
            scalars = payload.get('values')
            if not isinstance(scalars, list) or len(scalars) != 1 or not _finite_scalar(scalars[0]):
                raise UnsupportedPlan('Only finite singleton inline values have an explicit interpretation')
            inline.append(name)
        else:
            raise UnsupportedPlan('Unsupported plaintext payload')
        encodes[name] = payload
        defined.add(name)
    sequence = plan['initialization'] + plan['execution']
    if [s.get('ordinal') for s in sequence] != list(range(len(sequence))):
        raise UnsupportedPlan('Original instruction ordinals must be consecutive and ordered')
    counts, cells, last, rotations = Counter(), Counter(), {}, Counter()
    for index, step in enumerate(plan['execution']):
        kind, inputs, output = step.get('op'), step.get('inputs', []), step.get('output')
        if (step.get('kind') != 'compute' or kind not in OPS or len(inputs) != OPS[kind]
                or step.get('place') != dict(kind='host', rank=0) or output in defined or output not in values
                or any(name not in defined for name in inputs)):
            raise UnsupportedPlan('Unknown operation or non-topological original plan')
        operands, out = [values[name] for name in inputs], values[output]
        if operands[0]['kind'] != 'ciphertext' or out['kind'] != 'ciphertext':
            raise UnsupportedPlan('Compute must start with and produce a ciphertext')
        _metadata_equation(step, operands, out)
        cells[(kind, tuple((v['level'], v['scale_log2'], v['components']) for v in operands),
               (out['level'], out['scale_log2'], out['components']))] += 1
        if kind == 'rotate':
            rotations[step['attrs']['steps'] % 65536] += 1
        defined.add(output)
        counts[kind] += 1
        for name in inputs:
            last[name] = index
    if defined != set(values):
        raise UnsupportedPlan('Original plan contains undefined values')
    finals = set(plan['final_outputs'])
    live = set(plan['external_inputs'])
    payload_bytes = lambda name: values[name]['components'] * 131072 * (values[name]['level']+1) * 8
    peak_live, peak_bytes = len(live), sum(map(payload_bytes, live))
    for index, step in enumerate(plan['execution']):
        live.add(step['output'])
        peak_live, peak_bytes = max(peak_live, len(live)), max(peak_bytes, sum(map(payload_bytes, live)))
        for name in set(step['inputs']):
            if last[name] == index and name not in finals:
                live.discard(name)
        if step['output'] not in last and step['output'] not in finals:
            live.discard(step['output'])
    gates = _source_gates(plan, inline, source_proof, source_math,
                          public_fixture=public_fixture, plan_path=plan_path)
    return dict(status='metadata_valid', variant=VARIANT, original_plan_modified=False,
        value_count=len(values), execution_count=len(plan['execution']), initialization_count=len(encodes),
        inline_scalar_count=len(inline), inline_scalar_ids=inline,
        bundle_encode_count=len(encodes)-len(inline), unique_blob_count=len(blobs),
        op_counts=dict(counts), bootstrap_count=counts['boot'], metadata_cells=len(cells),
        value_levels=sorted({v['level'] for v in values.values()}),
        value_scales=sorted({v['scale_log2'] for v in values.values()}),
        rotation_steps=sorted(rotations), rotation_execution_counts=dict(rotations),
        last_use_ciphertext_peak_count=peak_live, last_use_ciphertext_peak_coefficient_bytes=peak_bytes,
        terminal_ciphertext_ids=sorted(live),
        eager_plaintext_coefficient_bytes=sum(131072*(values[name]['level']+1)*8 for name in encodes),
        lazy_plaintext_peak_coefficient_bytes=max((131072*(values[name]['level']+1)*8 for name in encodes), default=0),
        input_metadata=[values[i] for i in plan['external_inputs']],
        output_metadata=[values[i] for i in plan['final_outputs']], source_gates=gates,
        execution_scope_ready=gates['execution_scope_ready'], security_acceptance=None,
        memory_scope='Static uint64 RNS coefficient payload only; excludes DFT, keys, NTT/scratch/allocator, Go metadata and I/O',
        extra_integrity_checks=0)


def require_execution_ready(inventory):
    if inventory.get('execution_scope_ready') is not True:
        raise UnsupportedPlan('; '.join(inventory.get('source_gates', {}).get('unsupported_reasons', [])))


def clear_replay(plan, packed_input, bundle_vectors):
    """Public small-probe oracle; bootstrap is explicitly mathematical identity."""
    size = len(packed_input)
    if not size or not all(_finite_scalar(v) for v in packed_input):
        raise UnsupportedPlan('Expected a finite nonempty public input')
    values = {plan['external_inputs'][0]: list(packed_input)}
    for step in plan['initialization']:
        payload = step['payload']
        if payload['kind'] == 'inline':
            if len(payload['values']) != 1 or not _finite_scalar(payload['values'][0]):
                raise UnsupportedPlan('Invalid scalar fixture')
            data = payload['values'] * size
        else:
            data = list(bundle_vectors[payload['content']])
        if len(data) != size or not all(_finite_scalar(v) for v in data):
            raise UnsupportedPlan('Invalid public fixture constant')
        values[step['output']] = data
    for step in plan['execution']:
        op, args = step['op'], [values[name] for name in step['inputs']]
        if op in ('mod_switch', 'rescale', 'relinearize', 'boot'):
            output = list(args[0])
        elif op == 'rotate':
            k = step['attrs']['steps'] % size
            output = args[0][k:] + args[0][:k]
        elif op in ('mul_cc', 'mul_cp'):
            output = [a*b for a,b in zip(*args)]
        elif op in ('add_cc', 'add_cp'):
            output = [a+b for a,b in zip(*args)]
        elif op == 'negate':
            output = [-v for v in args[0]]
        else:
            raise UnsupportedPlan('Unsupported public fixture operation')
        if not all(_finite_scalar(v) for v in output):
            raise UnsupportedPlan('Nonfinite public fixture output')
        values[step['output']] = output
    return values[plan['final_outputs'][0]]


def make_public_fixture(*, bootstrap=False):
    """Two independent eleven-op fixtures; their eight-value periods repeat."""
    descriptors = []
    initial, execution = [], []
    def value(name, level, scale, components=2, kind='ciphertext'):
        descriptors.append(dict(id=name, kind=kind, place=dict(kind='host', rank=0),
            context=CONTEXT, ntt=True, level=level, scale_log2=scale, components=components))
        return name
    def plain(name, level, scale, payload):
        value(name, level, scale, 1, 'plaintext')
        initial.append(dict(kind='encode', output=name, payload=payload))
    def compute(kind, output, inputs, level, scale, components=2, **attrs):
        value(output, level, scale, components)
        execution.append(dict(kind='compute', op=kind, inputs=inputs, output=output,
                              place=dict(kind='host', rank=0), **({'attrs':attrs} if attrs else {})))
    value('x',16,40)
    raw = [(.05 + .1*i) for i in range(8)]
    vectors = {}
    if bootstrap:
        plain('one',3,11,dict(kind='inline',values=[1.0]))
        plain('offset',15,51,dict(kind='inline',values=[.125]))
        compute('mod_switch','a',['x'],3,40,target_level=3)
        compute('mul_cp','b',['a','one'],3,51)
        compute('boot','c',['b'],16,51,implementation='native',operator_profile=BOOT_PROFILE,
                target_level=16,target_scale_log2=51,target_components=2)
        compute('rotate','d',['c'],16,51,steps=1)
        compute('mul_cc','e',['d','d'],16,102,3)
        compute('relinearize','f',['e'],16,102)
        compute('rescale','g',['f'],15,51,target_level=15,target_scale_log2=51)
        compute('add_cp','h',['g','offset'],15,51)
        compute('mod_switch','i',['c'],15,51,target_level=15)
        compute('add_cc','j',['h','i'],15,51)
        compute('negate','final',['j'],15,51)
        expected = [-(raw[(i+1)%8]**2 + .125 + raw[i]) for i in range(8)]
    else:
        plain('half',5,87,dict(kind='bundle',content='public:half'))
        vectors['public:half'] = [.5]*8
        plain('offset',2,50,dict(kind='inline',values=[.125]))
        compute('mod_switch','a',['x'],5,40,target_level=5)
        compute('mul_cp','b',['a','half'],5,127)
        compute('rescale','c',['b'],4,76,target_level=4,target_scale_log2=76)
        compute('rotate','d',['c'],4,76,steps=1)
        compute('mul_cc','e',['d','d'],4,152,3)
        compute('relinearize','f',['e'],4,152)
        compute('rescale','g',['f'],3,101,target_level=3,target_scale_log2=101)
        compute('rescale','h',['g'],2,50,target_level=2,target_scale_log2=50)
        compute('add_cp','i',['h','offset'],2,50)
        compute('negate','j',['i'],2,50)
        compute('rotate','final',['j'],2,50,steps=-1)
        expected = [-(.25*v*v + .125) for v in raw]
    for ordinal, step in enumerate(initial+execution):
        step['ordinal'] = ordinal
    plan = dict(format_version=1,plan_id='1',target=dict(target_id='dacapo-own-native-public-fixture'),
        values=descriptors, external_inputs=['x'], initialization=initial, execution=execution,
        finalization=[], final_outputs=['final'], plaintext_bundle=dict(id='public-dacapo-fixture',version=1))
    manifest = dict(bundle_format_version=1,bundle_id='public-dacapo-fixture',version=1,
        blobs=[dict(content=name,byte_length=65536*8) for name in vectors])
    return plan, manifest, raw, vectors, expected


def write_public_fixture(work, *, bootstrap=False):
    """Prepare only public files; this function invokes no Go or cryptography."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    plan, manifest, raw, vectors, expected = make_public_fixture(bootstrap=bootstrap)
    bundle = work/'bundle'
    (bundle/'data').mkdir(parents=True)
    for name, data in vectors.items():
        repeated = data * (65536//len(data))
        (bundle/'data'/content_filename(name)).write_bytes(struct.pack('<65536d',*repeated))
    (work/'input.bin').write_bytes(struct.pack('<65536d',*(raw*(65536//len(raw)))))
    for name,data in [('plan.json',plan),('bundle/manifest.json',manifest),
        ('expected.json',dict(indices=list(range(8)),values=expected,
            scope='Public8-value period tiled to65536; independent scalar formula; no original-model acceptance'))]:
        (work/name).write_text(json.dumps(data,indent=2)+'\n')
    return dict(plan=str(work/'plan.json'),bundle=str(bundle),input=str(work/'input.bin'),expected=str(work/'expected.json'),
                fixture_operations=11,bootstrap_count=int(bootstrap),keygen_runs=0,ckks_runs=0)


def manifest_from_closed_archive(plan, archive_manifest):
    """Reconstruct locator/size metadata only; never open compressed data."""
    bundles = [item for item in archive_manifest['files'] if item['kind']=='runtime_bundle']
    if len(bundles)!=1:
        raise UnsupportedPlan('Expected exactly one closed bundle inventory')
    records=[]
    for item in bundles[0]['inventory']:
        if item['kind']=='file' and item['path'].endswith('.bin'):
            records.append(dict(content='sha256:'+Path(item['path']).stem,byte_length=item['bytes']))
    return dict(bundle_format_version=1,bundle_id=plan['plaintext_bundle']['id'],version=1,blobs=records,
                metadata_source='Archive inventory; original manifest content not re-opened')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    sources=parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--manifest',type=Path)
    sources.add_argument('--archive-manifest',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    plan=json.loads(args.plan.read_text())
    manifest=(json.loads(args.manifest.read_text()) if args.manifest else
              manifest_from_closed_archive(plan,json.loads(args.archive_manifest.read_text())))
    inventory=plan_inventory(plan,manifest,plan_path=args.plan)
    args.out.write_text(json.dumps(dict(native_parameters=native_parameters(),inventory=inventory,
        keygen_runs=0,crypto_runs=0,model_runs=0,go_commands=0,source_plan=str(args.plan)),indent=2)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
