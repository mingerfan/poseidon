"""Explicit logical inputs versus one trusted client-encrypted zero argument."""
from seal_artifact_gate import require

ZERO_NAME = 'zero_ct'
ZERO_ARGUMENT = {'dsl_name': ZERO_NAME, 'kind': 'encrypted_zero', 'source': 'trusted_client', 'slot_period': 4}


def has_zero_argument(layout):
    if 'execution_abi' in layout:
        from packed_input_abi import validate_layout
        validate_layout(layout)
        return 'auxiliary_ciphertexts' in layout
    if 'auxiliary_ciphertexts' not in layout:
        return False
    entries = layout['auxiliary_ciphertexts']
    require(type(entries) is list and len(entries) == 1 and type(entries[0]) is dict and
            type(entries[0].get('slot_period')) is int and entries[0] == ZERO_ARGUMENT,
            'Invalid encrypted-zero declaration; auxiliary input is not user-configurable data')
    return True


def logical_input_names(layout):
    if 'execution_abi' in layout:
        from packed_input_abi import validate_layout
        validate_layout(layout)
        return ('x',)
    if 'inputs' not in layout:
        return ('x',)
    specs = layout['inputs']
    require(type(specs) is list and 2 <= len(specs) <= 4, 'Invalid logical input manifest')
    names = tuple(s.get('dsl_name') if type(s) is dict else None for s in specs)
    require(names == ('x', 'y', 'z', 't')[:len(specs)], 'Noncanonical logical input order')
    return names


def physical_input_names(layout):
    names = logical_input_names(layout)
    return (*names, ZERO_NAME) if has_zero_argument(layout) else names


def execution_options(layout):
    return dict(expected_inputs=len(physical_input_names(layout)),
                logical_inputs=len(logical_input_names(layout)),
                encrypted_zero_input=has_zero_argument(layout),**artifact_options(layout))


def artifact_options(layout):
    if 'execution_abi' not in layout:
        return {}
    from packed_input_abi import gate_options
    return gate_options(layout)
