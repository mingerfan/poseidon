"""Saved complete complex outputs; execution, compiler and math proofs are external."""
import copy
import importlib.util
import math
from pathlib import Path
import re
import struct

from delivery_component import bound, require, sha
from poseidon_artifact_gate import PRIMES, CONTEXT
from poseidon_input_groups import read, plan, portion

PROFILE = 'poseidon-ordinary-complete-output-reader-profile-v1'
GROUPS = 'poseidon-ordinary-complete-output-groups-v1'
RESULT = 'poseidon-ordinary-complete-output-read-v1'
EXECUTION = 'poseidon-ordinary-cpu-complete-complex-execution-v1'
MANIFEST = 'complete-output-groups.json'
SLOTS = 16384
VALIDATOR = Path('/home/lhohy/Code/Poseidon/DSL/DSL_agent/validation/backend_certification/ordinary_complete_client_protocol_v1.py')


def freeze_profile(compiler, source, protocol):
    """Freeze an optional saved-output reader without selecting a native launcher."""
    require(compiler.get('format') == 'poseidon-compiler-managed-external-evaluation-v1', 'External compiler profile')
    require(source.get('format') == 'poseidon-B-independent-ordinary-client-source-profile-v1'
            and source.get('context_id') == compiler['configuration']['parameter_identity'] == CONTEXT
            and source.get('degree') == 32768 and source.get('security') == 'tc128'
            and source.get('q') == list(PRIMES[:13]) and source.get('p') == list(PRIMES[13:])
            and source.get('ordinary_temporary_key_sets_per_process') == 1
            and source.get('persistent_key_epochs') == 0, 'Original ordinary client context')
    require(protocol.get('format') == 'poseidon-B-ordinary-complete-client-live-protocol-requirements-v1'
            and protocol.get('normal_mode_output_ABI') == EXECUTION
            and protocol.get('physical_complex_slots_per_final') == SLOTS
            and protocol.get('maximum_cases_per_process') == 16
            and protocol.get('collector_role') == 'ordinary_cpu_acceptance_client'
            and protocol.get('source_profile_binding') == source['binding'], 'Published complete-output interface')
    return bound(dict(format=PROFILE, selector=copy.deepcopy(compiler['selector']),
        compiler_profile_binding=compiler['binding'], source_profile_binding=source['binding'],
        protocol_requirements_binding=protocol['binding'], client_build=copy.deepcopy(source['actual_build_artifact']),
        output_ABI=EXECUTION, execution_abi='unified-capacity-chunked-inputs-v4',
        maximum_cases_per_process=16, physical_complex_slots_per_final=SLOTS,
        q=list(PRIMES[:13]), p=list(PRIMES[13:]), metadata_validator=str(VALIDATOR),
        metadata_validator_source=dict(path=str(VALIDATOR),sha256=sha(VALIDATOR.read_bytes())),
        native_launcher_selected=False, native_execution_or_first_whole_SDK_certified=False))


def validate_profile(profile):
    require(type(profile) is dict and profile.get('format') == PROFILE
            and profile.get('output_ABI') == EXECUTION
            and profile.get('execution_abi') == 'unified-capacity-chunked-inputs-v4'
            and profile.get('physical_complex_slots_per_final') == SLOTS
            and profile.get('maximum_cases_per_process') == 16
            and profile.get('q') == list(PRIMES[:13]) and profile.get('p') == list(PRIMES[13:])
            and profile.get('metadata_validator') == str(VALIDATOR)
            and profile.get('native_launcher_selected') is False
            and profile.get('native_execution_or_first_whole_SDK_certified') is False, 'Optional saved complete-output profile')
    return profile


def validate_groups(payload, manifest):
    """Bind the complete ordered case range to the original per-process limit."""
    expected = plan(payload)
    require(type(manifest) is dict and set(manifest) == {'format','inputs','groups'}
            and manifest['format'] == GROUPS and type(manifest['inputs']) is int
            and manifest['inputs'] == expected['inputs'] and type(manifest['groups']) is list
            and len(manifest['groups']) == len(expected['groups']), 'Complete complex-output groups')
    nonces = set()
    for row, actual in zip(expected['groups'], manifest['groups']):
        require(type(actual) is dict and set(actual) == set(row)|{'run_nonce','input_sha256'}
                and all(type(actual[k]) is type(v) and actual[k] == v for k,v in row.items()), 'Ordered original native input group')
        nonce = actual['run_nonce']
        require(type(nonce) is str and re.fullmatch('[0-9a-f]{32}',nonce) and nonce not in nonces, 'Distinct observer nonce per group')
        require(type(actual['input_sha256']) is str and re.fullmatch('[0-9a-f]{64}',actual['input_sha256']), 'Frozen native group input descriptor')
        nonces.add(nonce)
    return expected


def complex_values(directory, descriptor):
    """Read all physical real/imaginary pairs with the published binary layout."""
    directory = Path(directory).absolute()
    require(directory.is_dir() and directory.resolve() == directory, 'Direct complete-output directory')
    require(type(descriptor) is dict and {'file','slots','bytes','encoding','component_order','sha256'} <= set(descriptor)
            and set(descriptor) <= {'file','slots','bytes','encoding','component_order','sha256','value_id'}, 'Physical output descriptor')
    name = descriptor['file']
    require(type(name) is str and name and name not in ('.','..') and Path(name).name == name
            and type(descriptor['slots']) is int and descriptor['slots'] == SLOTS
            and type(descriptor['bytes']) is int and descriptor['bytes'] == SLOTS*16
            and descriptor['encoding'] == 'complex_binary64_le_interleaved'
            and descriptor['component_order'] == ['real','imag'], 'Complete binary complex output layout')
    path = directory/name
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == SLOTS*16, 'Complete physical output extent')
    raw = path.read_bytes()
    require(len(raw) == SLOTS*16, 'Complete binary complex values')
    values = list(struct.iter_unpack('<dd', raw))
    require(len(values) == SLOTS and all(math.isfinite(x) and math.isfinite(y) for x,y in values), 'Finite complete real/imaginary cells')
    return values


def _metadata_checker():
    # Select this fixed B-owned interface only for the optional saved-output path.
    spec = importlib.util.spec_from_file_location('poseidon_complete_client_metadata_v1', VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_report


def read_groups(profile, output, request):
    """Decode saved groups for an external evaluator; do not assert native success."""
    validate_profile(profile)
    output = Path(output).absolute()
    require(output.is_dir() and output.resolve() == output, 'Direct complete-output root')
    payload = read(output/'inputs.json')
    manifest = read(output/MANIFEST)
    expected = validate_groups(payload, manifest)
    layout = request['layout']
    require(type(payload) is dict and set(payload) == {'format','execution_abi','input_slot_period','decode_slots','input_ciphertexts','output_ciphertexts','samples'}
            and payload['format'] == 'poseidon-agent-inputs-v2'
            and payload['execution_abi'] == layout.get('execution_abi') == profile['execution_abi'], 'Original wide input protocol')
    period = layout['input_slot_period']
    require(type(period) is int and period in (512,1024,2048,4096)
            and type(payload['input_slot_period']) is int and payload['input_slot_period'] == period
            and type(payload['decode_slots']) is int and payload['decode_slots'] == period, 'Original wide input period')
    count = payload['input_ciphertexts']
    require(type(count) is int and 2 <= count <= 17, 'Original physical input count')
    require(all(type(sample) is list and len(sample) == count
                and all(type(vector) is list and len(vector) == period
                        and all(type(v) in (int,float) and math.isfinite(v) for v in vector) for vector in sample)
                for sample in payload['samples']), 'Complete original input shapes')
    compiled = read(output/'bound/plan.json')
    returns = compiled['final_outputs']
    arguments = compiled['external_inputs']
    require(type(returns) is list and 1 <= len(returns) <= 16 and len(set(returns)) == len(returns)
            and all(type(v) is str and v for v in returns)
            and type(arguments) is list and len(arguments) == count and len(set(arguments)) == count
            and all(type(v) is str and v for v in arguments)
            and type(payload['output_ciphertexts']) is int
            and payload['output_ciphertexts'] == layout['output_ciphertexts'] == len(returns), 'Compiled physical return binding')
    selectors = layout['output_selectors']
    require(type(selectors) is list and selectors and all(type(s) is list and len(s) == 2
            and type(s[0]) is int and 0 <= s[0] < len(returns) and type(s[1]) is int and 0 <= s[1] < SLOTS for s in selectors), 'Declared complete-output selectors')
    check = _metadata_checker()
    rows = []
    physical = []
    for group, original in zip(manifest['groups'], expected['groups']):
        directory = output/group['directory']
        require(directory.is_dir() and directory.resolve() == directory, 'Direct complete-output group')
        require(read(directory/'inputs.json') == portion(payload, original), 'Complete original group inputs and order')
        require(read(directory/'terminal.json') == dict(index=group['index'],status='returned',returncode=0), 'Successful retained group terminal')
        metadata = check(read(directory/'report.json'),nonce=group['run_nonce'],case_count=group['stop']-group['start'],
            input_sha256=group['input_sha256'],period=period,input_ciphertexts=count,final_value_ids=returns,q=profile['q'],p=profile['p'])
        for local in range(group['stop']-group['start']):
            selected_real = [None]*len(selectors)
            selected_imag = [None]*len(selectors)
            for ordinal,value_id in enumerate(returns):
                descriptor = metadata['expected_files'][local*len(returns)+ordinal]
                descriptor = {k:v for k,v in descriptor.items() if k != 'case_index'}
                values = complex_values(directory,descriptor)
                for index,(ciphertext,slot) in enumerate(selectors):
                    if ciphertext == ordinal:selected_real[index],selected_imag[index] = values[slot]
                physical.append(dict(case=group['start']+local,final_ordinal=ordinal,value_id=value_id,
                    file=str((directory/descriptor['file']).relative_to(output)),complex_cells=SLOTS,
                    max_absolute_real=max(abs(x) for x,y in values),max_absolute_imag=max(abs(y) for x,y in values),
                    RMS_imag=math.hypot(*(y for x,y in values))/math.sqrt(SLOTS)))
            require(all(type(x) is float for x in selected_real+selected_imag), 'Complete declared real and imaginary decode')
            rows.append(dict(index=group['start']+local,CKKS_real=selected_real,CKKS_imag=selected_imag))
    require(len(rows) == manifest['inputs'], 'Every original complete-output input required')
    return bound(dict(format=RESULT,profile_binding=profile['binding'],input_groups=len(rows),
        native_process_groups=len(manifest['groups']),group_sizes=[g['stop']-g['start'] for g in manifest['groups']],
        physical_finals=len(physical),physical_complex_cells=len(physical)*SLOTS,inputs=rows,physical=physical,
        all_physical_cells_decoded=True,metadata_and_semantic_group_mapping_consistent=True,
        native_execution_provenance_verified=False,independent_compiler_math_or_application_verified=False,
        first_whole_model_or_SDK_qualified=False,reference_or_tolerance_changed=False))
