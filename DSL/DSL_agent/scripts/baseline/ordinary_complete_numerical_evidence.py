"""Match full-complex saved cells to frozen math; native provenance stays separate."""
import copy
from pathlib import Path

from delivery_component import bound, canonical, require, sha
from ordinary_complete_output import RESULT, read_groups, validate_profile

POLICY_FIELD = 'native_output_profile'


class CompleteNativeProofUnavailable(ValueError):
    """The ordinary-client source/build/input/terminal proof adapter is pending."""


def select_policy(policy, output_profile):
    validate_profile(output_profile)
    require(policy.get('format') == 'poseidon-first-output-numeric-policy-v1'
            and policy.get('independent_math') == dict(atol=1e-12,rtol=1e-12)
            and policy.get('CKKS') == dict(atol=1e-5,rtol=1e-4), 'Original frozen numerical policy')
    require(POLICY_FIELD not in policy, 'Select the complete-output profile once before generation')
    selected = copy.deepcopy(policy)
    selected.pop('binding',None)
    selected[POLICY_FIELD] = copy.deepcopy(output_profile)
    return bound(selected)


def join_rows(policy, source, physical_inputs, decoded, mathematical_rows):
    """Keep original/reference/DSL rows and attach only decoded native cells."""
    profile = validate_profile(policy[POLICY_FIELD])
    require(type(source) is bytes and decoded.get('format') == RESULT
            and decoded.get('profile_binding') == profile['binding']
            and decoded.get('native_execution_provenance_verified') is False
            and decoded.get('independent_compiler_math_or_application_verified') is False
            and decoded.get('first_whole_model_or_SDK_qualified') is False,
            'Saved-output decode with independent provenance scope')
    cells = decoded['inputs']
    require(type(physical_inputs) is list and type(cells) is list
            and len(physical_inputs) == len(cells) == decoded['input_groups'], 'Complete native input/output scope')
    observed = {}
    for index,(sample,cell) in enumerate(zip(physical_inputs,cells)):
        require(type(cell) is dict and cell.get('index') == index, 'Ordered native decoded input')
        key = sha(canonical(sample))
        require(key not in observed, 'Distinct frozen native inputs')
        observed[key] = cell
    require(set(observed) == set(policy['inputs']), 'Every frozen native input required')
    require(type(mathematical_rows) is list and len(mathematical_rows) == len(policy['inputs'])
            and len({r['input_sha256'] for r in mathematical_rows}) == len(mathematical_rows)
            and {r['input_sha256'] for r in mathematical_rows} == set(policy['inputs']), 'Complete independent mathematical input rows')
    rows = copy.deepcopy(mathematical_rows)
    for row in rows:
        require(row.get('source_sha256') == sha(source), 'Independent row uses the first frozen DSL')
        cell = observed[row['input_sha256']]
        require(type(cell.get('CKKS_real')) is list and type(cell.get('CKKS_imag')) is list
                and len(cell['CKKS_real']) == len(cell['CKKS_imag']) == len(row['declared_reference']), 'Declared complex logical output dimensions')
        row['CKKS_real'] = copy.deepcopy(cell['CKKS_real'])
        row['CKKS_imag'] = copy.deepcopy(cell['CKKS_imag'])
    return dict(inputs=rows,intermediate_decryptions=0,reference_or_tolerance_changed=False,
        native_output_profile=copy.deepcopy(profile),complete_output_read=copy.deepcopy(decoded))


def read_rows(request, source, proof, policy, mathematical_rows):
    from first_output_numerical_evidence import native_input_samples
    from first_output_backend import _json
    output = Path(proof['evidence'])/'attempt-00/output'
    decoded = read_groups(policy[POLICY_FIELD],output,request)
    inputs = native_input_samples(request,_json(output/'inputs.json'))
    return join_rows(policy,source,inputs,decoded,mathematical_rows)


def verify_rows(witness, request, source, policy, proof):
    """Formal dispatch preserves missing B native proof as an evaluation failure."""
    require(type(policy) is dict and POLICY_FIELD in policy
            and witness.get(POLICY_FIELD) == policy[POLICY_FIELD], 'Output decoder selection frozen before generation')
    require(proof['real_compilation_observed'] and proof['real_encrypted_execution_observed'], 'Real compiler and native proof required')
    expected = read_rows(request,source,proof,policy,witness['inputs'])
    require(witness.get('complete_output_read') == expected['complete_output_read'], 'Complete saved physical observation changed')
    from first_output_backend import _json
    report = _json(Path(proof['evidence'])/'report.json')
    comparison = report['attempts'][0].get('comparison',{})
    rows = witness['inputs']
    from first_output_numerical_evidence import native_input_samples
    native_inputs = native_input_samples(request,_json(Path(proof['evidence'])/'attempt-00/output/inputs.json'))
    references = {row['input_sha256']:row['declared_reference'] for row in expected['inputs']}
    cells = expected['complete_output_read']['inputs']
    require(comparison.get('actual') == [r['CKKS_real'] for r in cells]
            and comparison.get('actual_imag') == [r['CKKS_imag'] for r in cells]
            and comparison.get('reference') == [references[sha(canonical(sample))] for sample in native_inputs], 'Original real/imaginary comparison and frozen reference')
    require(all(row['CKKS_real'] == actual['CKKS_real'] and row.get('CKKS_imag') == actual['CKKS_imag']
                for row,actual in zip(rows,expected['inputs'])), 'Reported cells differ from the complete native files')
    require(type(witness.get('intermediate_decryptions')) is int and witness['intermediate_decryptions'] == 0
            and witness.get('reference_or_tolerance_changed') is False, 'Original final-only execution and reference boundary')
    # The published B reader v2 proves host-record consistency only. Source,
    # build, terminal, group-input and process-local key provenance are pending.
    # Register its concrete complete execution reader in a later frozen version.
    raise CompleteNativeProofUnavailable('Ordinary complete-client execution proof adapter pending: source/build/terminal/input/process-local-key evidence')
