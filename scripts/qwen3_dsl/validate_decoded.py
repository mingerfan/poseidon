#!/usr/bin/env python3
"""Check every decoded Qwen output against independent exact and polynomial references."""
import argparse
import json
from pathlib import Path

import numpy as np

from export_runner_fixture import (ATOL, RTOL, map_outputs, verify_provenance,
                                   verify_signature, validate_array)
from qwen3.weights import file_hash


def metrics(actual, expected):
    error = actual.real - expected
    bad_real = int(np.count_nonzero(np.abs(error) > ATOL + RTOL * np.abs(expected)))
    bad_imaginary = int(np.count_nonzero(np.abs(actual.imag) > ATOL))
    denominator = np.linalg.norm(actual.real) * np.linalg.norm(expected)
    return dict(count=int(actual.size), max_abs_real=float(np.max(np.abs(error))),
                rms_real=float(np.sqrt(np.mean(error**2))),
                max_abs_complex=float(np.max(np.abs(actual - expected))),
                max_imaginary=float(np.max(np.abs(actual.imag))),
                cosine_real=(float(np.dot(actual.real.ravel(), expected.ravel()) / denominator)
                             if denominator != 0 else None),
                bad_real=bad_real, bad_imaginary=bad_imaginary,
                passed=bad_real == 0 and bad_imaginary == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signature', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--runner-fixture', type=Path, required=True)
    parser.add_argument('--decoded', type=Path, required=True)
    parser.add_argument('--runner-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    signature = read(args.signature)
    provenance = read(args.fixture / 'provenance.json')
    configuration = read(args.fixture / 'model.json')
    verify_provenance(provenance, configuration)
    verify_signature(signature, configuration, provenance['positions'])
    runner_fixture, decoded, runner_report = map(read, (
        args.runner_fixture, args.decoded, args.runner_report))
    fixture_digest = file_hash(args.runner_fixture)
    model = runner_fixture['model']
    if (runner_fixture['format_version'] != 1 or
            runner_fixture['context']['poly_degree'] != signature['slots'] * 2 or
            model['signature_sha256'] != file_hash(args.signature) or
            model['weights_manifest_sha256'] != signature['weights_binding'] or
            any(model.get(key) != value for key, value in provenance.items())):
        raise ValueError('Runner fixture model/signature provenance does not match reference artifacts')
    if runner_fixture['tolerances'] != dict(atol=ATOL, rtol=RTOL):
        raise ValueError('Runner tolerance differs from the predefined acceptance tolerance')
    if (decoded['format_version'] != 1 or
            decoded['fixture_sha256'] != fixture_digest or
            decoded['slots_per_output'] != signature['slots']):
        raise ValueError('Decoded output is not bound to the matching fixture and slot capacity')
    if (runner_report['format_version'] != 1 or
            runner_report['fixture_sha256'] != fixture_digest or
            runner_report['input_count'] != 2 or runner_report['output_count'] != 261 or
            runner_report['slots_per_output'] != signature['slots'] or
            runner_report['context'] != runner_fixture['context'] or
            runner_report['tolerances'] != runner_fixture['tolerances'] or
            any(runner_report['model'].get(key) != value for key, value in model.items()
                if key != 'full_model_ckks_validated')):
        raise ValueError('Runner report is not bound to the matching fixture/context')
    rows = decoded['outputs']
    if (len(rows) != 261 or len(rows) != len(signature['outputs']) or
            len(runner_report['outputs']) != len(rows)):
        raise ValueError('Incomplete output: 149 logits and 112 KV ciphertexts required')
    actual = []
    tail_max, bad_tail = 0., 0
    for row, description, reported in zip(rows, signature['outputs'], runner_report['outputs']):
        if row['name'] != description['name'] or row['prefix_size'] != description['prefix_size']:
            raise ValueError('Output name/order/prefix does not match the traced full model')
        if (reported['name'] != row['name'] or reported['prefix_size'] != row['prefix_size'] or
                type(row['value_id']) is not int or row['value_id'] < 0 or
                reported['value_id'] != row['value_id']):
            raise ValueError('Decoded output ValueId/order does not match runner report')
        real, imaginary = np.asarray(row['real']), np.asarray(row['imaginary'])
        validate_array(real, (signature['slots'],), row['name'] + ' real')
        validate_array(imaginary, (signature['slots'],), row['name'] + ' imaginary')
        prefix = description['prefix_size']
        actual.append(real[:prefix] + 1j * imaginary[:prefix])
        tail_max = max(tail_max, float(np.max(np.abs(real[prefix:]))),
                       float(np.max(np.abs(imaginary[prefix:]))))
        bad_tail += int(np.count_nonzero(np.abs(real[prefix:]) > ATOL))
        bad_tail += int(np.count_nonzero(np.abs(imaginary[prefix:]) > ATOL))
    result = dict(format_version=1, tolerance=dict(atol=ATOL, rtol=RTOL),
                  model=provenance['model'], revision=provenance['revision'],
                  output_count=261, slots_per_output=signature['slots'],
                  references={}, max_tail_abs=tail_max, bad_tail_components=bad_tail,
                  artifacts={field: file_hash(getattr(args, field)) for field in (
                      'signature', 'runner_fixture', 'decoded', 'runner_report')})
    for kind in ('exact', 'approximate'):
        path = args.fixture / ('reference-' + kind + '.npz')
        if model['reference_' + kind + '_sha256'] != file_hash(path):
            raise ValueError('Reference artifact hash does not match runner fixture: ' + kind)
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {'logits', 'keys', 'values'}:
                raise ValueError('Reference must provide complete logits/keys/values arrays')
            reference = {key: archive[key] for key in archive.files}
        validate_array(reference['logits'], (151936,), kind + ' logits')
        validate_array(reference['keys'], (28, 2, 1024), kind + ' keys')
        validate_array(reference['values'], (28, 2, 1024), kind + ' values')
        expected_rows = map_outputs(signature['outputs'], reference, provenance['positions'])
        if kind == 'approximate' and runner_fixture['expected_outputs'] != expected_rows:
            raise ValueError('Runner expected outputs differ from the independent approximate reference')
        expected = [np.asarray(row['values']) for row in expected_rows]
        logits, expected_logits = np.concatenate(actual[:149]), np.concatenate(expected[:149])
        kv, expected_kv = np.concatenate(actual[149:]), np.concatenate(expected[149:])
        if logits.size != 151936 or kv.size != 28 * 2 * 2 * 1024:
            raise ValueError('Decoded logit/KV element counts differ from complete model')
        item = dict(reference_sha256=file_hash(path), logits=metrics(logits, expected_logits),
                    kv=metrics(kv, expected_kv), decoded_top1=int(np.argmax(logits.real)),
                    reference_top1=int(np.argmax(expected_logits)))
        item['top1_match'] = item['decoded_top1'] == item['reference_top1']
        result['references'][kind] = item
    result['runner_completed'] = runner_report.get('stage') == 'complete'
    result['runner_passed'] = runner_report.get('passed') is True
    result['passed'] = (result['runner_completed'] and result['runner_passed'] and bad_tail == 0
                        and all(item['logits']['passed'] and item['kv']['passed']
                                and item['top1_match'] for item in result['references'].values()))
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
