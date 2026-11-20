"""Trusted three-layer mathematical records for the external assessment only.

References come from the original mathematical graph, a retained hash-bound
dual-reference qualification, or the unchanged native-source reference worker's
record. None come from CKKS outputs. The DSL is interpreted as checked AST data.
There is no Provider, compiler, encryption, decryption, execution or repair call.
"""
import copy
from pathlib import Path
import sys
import numpy as np

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha, write_once
from first_output_backend import _json, verify_qualification_binding
from first_output_dsl_math import evaluate as dsl_mathematics

BASE = Path('/home/lhohy/Code/Poseidon/DSL/DSL_agent/scripts/baseline')
FORMAT = 'poseidon-three-layer-mathematical-evidence-v1'


def implementation_bundle(extra_sources=()):
    """Conservative existing source closure plus this new mathematical mechanism."""
    from application_identity import source_closure, DYNAMIC
    from first_output_dsl_math import __file__ as dsl_source
    authorities = dict(DYNAMIC)
    authorities[('source_precision_reference.py', 'importlib.import_module(module_name)')] = (
        'unchanged readonly native-source worker; this adapter only checks its frozen retained record')
    closure = source_closure(BASE, ('decorated_functions.py', 'operator_arithmetic_graph.py',
        'operator_reference.py', 'benchmark_math.py', 'program_execution.py',
        'unified_public_contract.py', 'source_precision_reference.py'), dynamic_authorities=authorities)
    sources = {str(BASE/name): digest for name, digest in closure['sources'].items() if name.endswith('.py')}
    assets = {str(BASE/name): digest for name, digest in closure['sources'].items() if not name.endswith('.py')}
    for p in (Path(__file__).absolute(), Path(dsl_source).absolute(), BASE/'delivery_component.py',
              BASE/'first_output_backend.py', BASE/'first_output_assessment.py',
              BASE/'first_output_numerical_evidence.py', *map(Path, extra_sources)):
        require(p.is_absolute() and p.suffix == '.py', 'Mathematical source file identity')
        sources[str(p)] = sha(read_bytes(p))
    return dict(sources=dict(sorted(sources.items())), assets=dict(sorted(assets.items())),
        import_edges=closure['import_edges'], dynamic_dependencies=closure['dynamic_dependencies'],
        external_imports=closure['external_imports'])


def implementation_sources(extra_sources=()):
    return implementation_bundle(extra_sources)['sources']


def _normalized(request, samples):
    from program_execution import normalize_samples, _pack_logical_samples
    logical = normalize_samples(request, samples)
    physical = _pack_logical_samples(request, logical).tolist()
    wire = [{name: value.tolist() for name, value in row.items()} for row in logical]
    return logical, physical, wire


def make_policy(request, original_model, samples, *, domain, approximation_requirements,
                application_requirements, whole_model_required=False, reference_kind='json_float64',
                retained_reference=None, native_source=None, application_contract=None,
                maximum_scalar_operations=2**28, extra_sources=()):
    """Freeze private evaluation inputs and producer identities before generation."""
    from first_output_assessment import evaluation_policy
    require(original_model['inputs'] == request['model']['inputs'], 'Original input signature differs')
    logical, physical, wire = _normalized(request, samples)
    require(type(domain) is dict and set(domain) == {i['name'] for i in original_model['inputs']}, 'Explicit original input domain')
    for interval in domain.values():
        require(type(interval) is list and len(interval) == 2 and all(type(x) in (int, float) and np.isfinite(x) for x in interval)
                and interval[0] <= interval[1], 'Finite input interval')
    require(reference_kind in ('json_float64', 'retained_core', 'native_source'), 'Original reference semantics')
    require(type(maximum_scalar_operations) is int and 1 <= maximum_scalar_operations <= 2**28, 'Frozen mathematical work budget')
    if reference_kind == 'retained_core':
        require(original_model == request['model'] and type(retained_reference) is dict,
                'Retained reference cannot replace an original source or declared approximation')
    else:
        require(retained_reference is None, 'Unexpected retained mathematical reference')
    if reference_kind == 'native_source':
        require(type(native_source) is dict and set(native_source) == {'manifest_sha256', 'input_dtypes', 'output_structure', 'source_identity'},
                'Frozen original native-source selection required')
    else:
        require(native_source is None, 'Unexpected original native-source selection')
    require(not application_requirements or type(application_contract) is dict,
            'Frozen application evidence contract required; sample agreement cannot replace dataset accuracy')
    implementation = implementation_bundle(extra_sources)
    sources = implementation['sources']
    spec = bound(dict(format='poseidon-frozen-mathematical-producer-v1', reference_kind=reference_kind,
        original_model_sha256=sha(canonical(original_model)), declared_model_sha256=sha(canonical(request['model'])),
        logical_input_sha256s=[sha(canonical(x)) for x in wire], input_domain=copy.deepcopy(domain),
        retained_reference=copy.deepcopy(retained_reference), native_source=copy.deepcopy(native_source),
        application_contract=copy.deepcopy(application_contract), maximum_scalar_operations=maximum_scalar_operations,
        producer_sources=sources, producer_assets=implementation['assets'],
        import_edges=implementation['import_edges'], dynamic_dependencies=implementation['dynamic_dependencies'],
        external_imports=implementation['external_imports'], python=sys.version, numpy=np.__version__))
    policy = evaluation_policy([sha(canonical(x)) for x in physical], domain_sha256=sha(canonical(domain)),
        approximation_requirements=approximation_requirements, application_requirements=application_requirements,
        whole_model_required=whole_model_required, independent_math_sources=sources,
        original_model_sha256=spec['original_model_sha256'])
    policy = {k: v for k, v in policy.items() if k != 'binding'}
    policy['mathematical_producer'] = spec
    return bound(policy)


def _check_policy(policy, request, original_model, samples):
    checked_bound(policy); spec = checked_bound(policy['mathematical_producer'])
    require(spec['format'] == 'poseidon-frozen-mathematical-producer-v1'
            and spec['original_model_sha256'] == policy['original_model_sha256'] == sha(canonical(original_model))
            and spec['declared_model_sha256'] == sha(canonical(request['model'])), 'Original/declared mathematical model changed')
    require(spec['producer_sources'] == policy['independent_math_sources']
            and all(sha(read_bytes(p)) == h for p, h in spec['producer_sources'].items())
            and all(sha(read_bytes(p)) == h for p, h in spec['producer_assets'].items())
            and spec['python'] == sys.version and spec['numpy'] == np.__version__, 'Mathematical producer identity changed')
    logical, physical, wire = _normalized(request, samples)
    require([sha(canonical(x)) for x in physical] == policy['inputs']
            and [sha(canonical(x)) for x in wire] == spec['logical_input_sha256s']
            and sha(canonical(spec['input_domain'])) == policy['frozen_domain_sha256'], 'Frozen mathematical inputs/domain changed')
    return spec, logical, physical, wire


def _flat(values, model):
    require(type(values) is dict and set(values) == {o['name'] for o in model['outputs']}, 'Mathematical named outputs')
    arrays = [np.asarray(values[o['name']], dtype=np.float64).reshape(-1) for o in model['outputs']]
    require(all(x.size and np.isfinite(x).all() for x in arrays), 'Finite mathematical named outputs')
    return np.concatenate(arrays).tolist()


def _reference(model, inputs, *, ideal):
    from operator_arithmetic_graph import FORMATS, evaluate_reference
    if model.get('format') in FORMATS:
        return evaluate_reference(model, inputs, ideal=ideal)
    from benchmark_math import evaluate
    return evaluate(model, inputs)


def _retained_values(selection, request, original_model, physical):
    require(original_model == request['model'] and type(selection) is dict
            and set(selection) == {'path', 'sha256'}, 'Retained core reference descriptor')
    path = Path(selection['path']).absolute()
    require(sha(read_bytes(path)) == selection['sha256'], 'Retained reference binding changed')
    proof = checked_bound(_json(path)); root = Path(proof['evidence'])
    verify_qualification_binding(proof, request, read_bytes(root/'attempt-00/candidate.py'))
    inputs = _json(root/'attempt-00/output/inputs.json')
    require(type(inputs) is list, 'Retained reference input protocol needs a certified decoder')
    mapping = {sha(canonical(x)): i for i, x in enumerate(inputs)}
    require(len(mapping) == len(inputs), 'Ambiguous retained reference inputs')
    with np.load(root/'arrays.npz', allow_pickle=False) as arrays:
        reference = np.array(arrays['reference'], dtype=np.float64, copy=True)
    require(reference.ndim == 2 and reference.shape[0] == len(inputs) and np.isfinite(reference).all(), 'Retained reference shape')
    require(reference.tolist() == _json(root/'report.json')['attempts'][0]['comparison']['reference'], 'Original retained reference comparison changed')
    return [reference[mapping[sha(canonical(x))]].tolist() for x in physical]


def _native_values(selection, descriptor, request, wire):
    require(type(descriptor) is dict and set(descriptor) == {'directory', 'request_sha256', 'report_sha256'},
            'Original native-source worker evidence required')
    root = Path(descriptor['directory']).absolute()
    require(sha(read_bytes(root/'request.json')) == descriptor['request_sha256']
            and sha(read_bytes(root/'native-reference.json')) == descriptor['report_sha256'], 'Original native-source evidence changed')
    from source_precision_reference import check_report, source_identity
    original_request = _json(root/'request.json')
    report = check_report(_json(root/'native-reference.json'), original_request)
    require(report['source_identity'] == selection['source_identity'] == source_identity()
            and report['manifest_sha256'] == selection['manifest_sha256']
            and report['input_dtypes'] == selection['input_dtypes']
            and report['output_structure'] == selection['output_structure']
            and canonical(original_request['samples']) == canonical(wire), 'Original native-source selection/input binding')
    from model_output_contract import decoded_tensor
    result = []
    for sample in report['outputs']:
        result.append(np.concatenate([np.asarray(decoded_tensor(sample[o['name']], original_request['output_shapes'][o['name']]),
            dtype=np.float64).reshape(-1) for o in request['model']['outputs']]).tolist())
    return result


def _application_values(contract, descriptor, policy, request, source):
    if not policy['application_requirements']:
        require(descriptor is None, 'Unexpected application record')
        return {}
    require(type(contract) is dict and type(descriptor) is dict and set(descriptor) == {'path', 'sha256'},
            'Independent application evidence required; never infer dataset accuracy from a few input rows')
    path = Path(descriptor['path']).absolute()
    require(sha(read_bytes(path)) == descriptor['sha256'], 'Application evidence changed')
    report = checked_bound(_json(path))
    require(report.get('format') == 'poseidon-first-output-application-evidence-v1'
            and report.get('contract_sha256') == sha(canonical(contract))
            and report.get('original_model_sha256') == policy['original_model_sha256']
            and report.get('declared_model_sha256') == sha(canonical(request['model']))
            and report.get('source_sha256') == sha(source)
            and report.get('samples') == contract['samples']
            and report.get('dataset_sha256') == contract['dataset_sha256']
            and report.get('producer_sources') == contract['producer_sources']
            and report.get('candidate_python_executed') is False
            and report.get('CKKS_outputs_used_as_reference') is False, 'Application dataset/model/source provenance')
    require(bool(contract['producer_sources']) and all(policy['independent_math_sources'].get(p) == h
            for p, h in contract['producer_sources'].items()), 'Unfrozen application producer')
    values = report.get('metrics')
    require(type(values) is dict and all(r['id'] in values and type(values[r['id']]) in (int, float)
            and np.isfinite(values[r['id']]) for r in policy['application_requirements']), 'Application metric evidence')
    return values


def produce_record(request, source, original_model, samples, policy, output, *,
                   native_source_evidence=None, application_evidence=None):
    """Produce mathematical evidence once; failure never edits or repairs the DSL."""
    path = Path(output).absolute()
    require(path.suffix == '.json' and path.parent.resolve() == path.parent, 'Direct mathematical record path')
    if path.exists():
        raise FileExistsError(path)
    require(type(source) is bytes, 'Exact original DSL bytes required')
    spec, logical, physical, wire = _check_policy(policy, request, original_model, samples)
    if spec['reference_kind'] == 'retained_core':
        originals = _retained_values(spec['retained_reference'], request, original_model, physical)
    elif spec['reference_kind'] == 'native_source':
        originals = _native_values(spec['native_source'], native_source_evidence, request, wire)
    else:
        require(native_source_evidence is None, 'Unexpected original source evidence')
        originals = None
    application = _application_values(spec['application_contract'], application_evidence, policy, request, source)
    rows = []
    for index, (inputs, packed) in enumerate(zip(logical, physical)):
        if spec['reference_kind'] == 'retained_core':
            original = declared = originals[index]
        else:
            declared = _flat(_reference(original_model, inputs, ideal=False), original_model)
            original = (originals[index] if originals is not None else
                declared if original_model == request['model'] else _flat(_reference(original_model, inputs, ideal=True), original_model))
        mathematical = dsl_mathematics(source, request, packed,
            maximum_scalar_operations=spec['maximum_scalar_operations'])
        require(len(original) == len(declared) == len(mathematical['values']), 'Whole mathematical output arity')
        domain = all(np.all(value >= spec['input_domain'][name][0]) and np.all(value <= spec['input_domain'][name][1])
                     for name, value in inputs.items())
        rows.append(dict(input_sha256=sha(canonical(packed)), source_sha256=sha(source), original_reference=list(original),
            declared_reference=list(declared), dsl_mathematical_output=mathematical['values'],
            inside_frozen_domain=bool(domain), domain_sha256=policy['frozen_domain_sha256'],
            application_metrics=copy.deepcopy(application), whole_model_output=True,
            mathematical_scalar_operations=mathematical['scalar_operations']))
    _check_policy(policy, request, original_model, samples)
    record = bound(dict(format=FORMAT, request_id=request['request_id'], source_sha256=sha(source),
        policy_sha256=sha(canonical(policy)), original_model_sha256=policy['original_model_sha256'],
        declared_model_sha256=sha(canonical(request['model'])), producer_sources=policy['independent_math_sources'],
        producer_spec_binding=spec['binding'], reference_origin=spec['reference_kind'],
        candidate_python_executed=False, CKKS_outputs_used_as_reference=False, encrypted_execution=False,
        native_source_evidence=copy.deepcopy(native_source_evidence), application_evidence=copy.deepcopy(application_evidence),
        test_only=False, inputs=rows))
    write_once(path, canonical(record))
    return dict(path=str(path), sha256=sha(read_bytes(path)))
