"""Validate the explicit encrypted-zero binding in saved real CPU evidence.

Distinct public ciphertext fingerprints detect accidental reuse; they are not a
proof of random-number-generator quality. No keys or plaintexts are read here.
"""
import re

from cipher_abi import execution_options, has_zero_argument
from seal_artifact_gate import require


def verify_zero_execution(layout, execution):
    enabled = has_zero_argument(layout)
    if not enabled:
        require('auxiliary_encrypted_zero' not in execution, 'Undeclared auxiliary ciphertext')
        return
    options = execution_options(layout)
    logical = options['logical_inputs']
    zero = execution.get('auxiliary_encrypted_zero')
    require(type(zero) is dict and set(zero) == {'logical_input_count', 'index', 'fresh_per_batch',
        'nontransparent', 'binding', 'ciphertext_fingerprints'}, 'Missing exact zero execution evidence')
    require(type(zero['logical_input_count']) is int and zero['logical_input_count'] == logical and
            type(zero['index']) is int and zero['index'] == logical and
            zero['fresh_per_batch'] is True and zero['nontransparent'] is True and
            zero['binding'] == 'trusted_client_public_key_encryption', 'Untrusted zero binding')
    require(execution.get('encrypted_execution') is True and execution.get('bootstrap_executed') is False and
            type(execution.get('input_batches')) is int and execution['input_batches'] == 4 and
            type(execution.get('encrypted_input_count')) is int and
            execution['encrypted_input_count'] == options['expected_inputs'], 'Wrong physical encrypted input count')
    fingerprints = zero['ciphertext_fingerprints']
    require(type(fingerprints) is list and len(fingerprints) == 4 and
            all(type(s) is str and re.fullmatch('[0-9a-f]{16}', s) for s in fingerprints) and
            len(set(fingerprints)) == 4, 'Auxiliary ciphertext not freshly encrypted for each input batch')
    observations = execution.get('ciphertext_metadata')
    require(type(observations) is list and len(observations) == 4, 'Missing native ciphertext metadata')
    for observation in observations:
        inputs = observation.get('inputs')
        require(type(inputs) is list and len(inputs) == options['expected_inputs'] and
                inputs[logical].get('polynomials') == 2, 'Missing native zero input metadata')
