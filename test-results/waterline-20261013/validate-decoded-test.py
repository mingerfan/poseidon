#!/usr/bin/env python3
"""Checker-only regression with simulated decoded slots; never CKKS acceptance."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(',', ':'), allow_nan=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--fixture-dir', type=Path, default=root / 'build-waterline-sync/checker-fixture')
    parser.add_argument('--checker', type=Path, default=root / 'scripts/qwen3_dsl/validate_decoded.py')
    args = parser.parse_args()
    source = args.fixture_dir.resolve()
    signature = json.loads((source / 'signature.json').read_text())
    original_fixture = json.loads((source / 'qwen3-runner-bsgs.json').read_text())
    assert len(signature['outputs']) == 261 and signature['slots'] == 8192
    assert len(original_fixture['expected_outputs']) == 261
    with tempfile.TemporaryDirectory(prefix='checker-test-only-') as directory:
        temporary = Path(directory)
        for name in ('signature.json', 'provenance.json', 'model.json',
                     'reference-exact.npz', 'reference-approximate.npz'):
            shutil.copyfile(source / name, temporary / name)
        fixture_path = temporary / 'runner-fixture.json'
        decoded_path = temporary / 'checker-test-only-decoded.json'
        report_path = temporary / 'checker-test-only-runner-report.json'
        fixture = copy.deepcopy(original_fixture)
        slots = signature['slots']
        decoded = dict(format_version=1, fixture_sha256='', slots_per_output=slots,
                       checker_test_only=True, outputs=[])
        report = dict(format_version=1, stage='complete', passed=True,
                      fixture_sha256='', input_count=2, output_count=261,
                      slots_per_output=slots, context=fixture['context'],
                      tolerances=fixture['tolerances'], model=copy.deepcopy(fixture['model']),
                      decoded_path=str(decoded_path), checker_test_only=True, outputs=[])
        for index, (description, expected) in enumerate(zip(signature['outputs'], fixture['expected_outputs'])):
            assert description['name'] == expected['name']
            prefix = description['prefix_size']
            # Synthetic ValueIds only exercise the existing report/decoded binding.
            value_id = 900000 + index
            decoded['outputs'].append(dict(name=description['name'], value_id=value_id,
                prefix_size=prefix, real=expected['values'] + [0.] * (slots-prefix), imaginary=[0.] * slots))
            report['outputs'].append(dict(name=description['name'], value_id=value_id,
                prefix_size=prefix, level=0, scale_log2=40, passed=True,
                slots_checked=slots, finite_slots=slots, bad_real_slots=0,
                bad_imaginary_slots=0, nonfinite_slots=0, bad_prefix_slots=0, bad_tail_slots=0,
                max_abs_error=0., max_relative_error_floor_atol=0., rms_error=0.,
                max_imaginary=0., max_tail_abs=0.))

        def bind():
            write_json(fixture_path, fixture)
            decoded['fixture_sha256'] = report['fixture_sha256'] = digest(fixture_path)
            report['model'] = copy.deepcopy(fixture['model'])
            report['model']['full_model_ckks_validated'] = True  # Production runner updates this field.

        environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        tests = 0

        def check(name, passed=False, error=None, predicate=None):
            nonlocal tests
            output = temporary / (name + '.json')
            write_json(decoded_path, decoded); write_json(report_path, report)
            process = subprocess.run([sys.executable, str(args.checker.resolve()),
                '--signature', str(temporary / 'signature.json'), '--fixture', str(temporary),
                '--runner-fixture', str(fixture_path), '--decoded', str(decoded_path),
                '--runner-report', str(report_path), '--output', str(output)],
                capture_output=True, text=True, env=environment, timeout=30)
            if passed:
                assert process.returncode == 0, process.stderr
                result = json.loads(output.read_text())
                assert result['passed'] is True and result['output_count'] == 261
                for reference in result['references'].values():
                    assert reference['logits']['count'] == 151936
                    assert reference['kv']['count'] == 28*2*2*1024
            else:
                assert process.returncode != 0, 'checker incorrectly accepted ' + name
                if error:
                    assert error in process.stderr, (name, process.stderr)
                    assert not output.exists(), 'structural failure wrote a successful artifact'
                else:
                    result = json.loads(output.read_text())
                    assert result['passed'] is False
                    if predicate: assert predicate(result), (name, result)
            tests += 1
            print('PASS checker-test-only', name, 'accepted' if passed else 'rejected', flush=True)

        bind()
        print('SCOPE checker-test-only: simulated 261 full-8192-slot outputs from independent '
              'reference prefixes and zero tails; no encryption, GPU, or model run.', flush=True)
        check('positive-all-logits-kv-and-tails', passed=True)
        original_outputs = decoded['outputs']
        decoded['outputs'] = [dict(row, real=[0.]*slots, imaginary=[0.]*slots) for row in original_outputs]
        check('finite-all-zero-decoded', predicate=lambda r:
              all(v[field]['bad_real'] > 0 and v[field]['cosine_real'] is None
                  for v in r['references'].values() for field in ('logits', 'kv')))
        decoded['outputs'] = original_outputs
        for name, row_index, component, element, category in (
            ('last-valid-logit-151935', 148, 'real', 383, 'logits'),
            ('last-kv-layer27-value-position1-element1023', 260, 'real', 1023, 'kv'),
            ('imaginary-prefix', 0, 'imaginary', 0, 'logits'),
            ('last-logit-real-tail-slot8191', 148, 'real', 8191, 'tail'),
            ('last-kv-imaginary-tail-slot8191', 260, 'imaginary', 8191, 'tail')):
            values = decoded['outputs'][row_index][component]
            old = values[element]
            values[element] += 10 * (.001 + .0001*abs(old))
            field = 'bad_imaginary' if component == 'imaginary' else 'bad_real'
            check(name, predicate=(lambda r: r['bad_tail_components'] > 0) if category == 'tail' else
                  (lambda r, category=category, field=field:
                   all(v[category][field] > 0 for v in r['references'].values())))
            values[element] = old
        removed = decoded['outputs'].pop()
        check('missing-last-output', error='Incomplete output'); decoded['outputs'].append(removed)
        decoded['outputs'][0], decoded['outputs'][1] = decoded['outputs'][1], decoded['outputs'][0]
        check('reordered-outputs', error='Output name/order/prefix')
        decoded['outputs'][0], decoded['outputs'][1] = decoded['outputs'][1], decoded['outputs'][0]
        old = decoded['fixture_sha256']; decoded['fixture_sha256'] = '0'*64
        check('stale-decoded-fixture-hash', error='Decoded output is not bound'); decoded['fixture_sha256'] = old
        old = report['fixture_sha256']; report['fixture_sha256'] = '0'*64
        check('stale-runner-report-fixture-hash', error='Runner report is not bound'); report['fixture_sha256'] = old
        report['outputs'][-1]['value_id'] += 1
        check('mismatched-runner-output-value-id', error='ValueId/order'); report['outputs'][-1]['value_id'] -= 1
        report['stage'] = 'execute_complete_plan'
        check('runner-not-complete', predicate=lambda r: r['runner_completed'] is False)
        report['stage'] = 'complete'; report['passed'] = False
        check('runner-reported-failure', predicate=lambda r: r['runner_passed'] is False); report['passed'] = True
        decoded['outputs'][-1]['imaginary'][-1] = float('nan')
        check('nonfinite-last-imaginary-slot', error='must be finite floating-point')
        decoded['outputs'][-1]['imaginary'][-1] = 0.
        original_hash = fixture['model']['signature_sha256']
        fixture['model']['signature_sha256'] = '0'*64; bind()
        check('stale-signature-binding', error='model/signature provenance')
        fixture['model']['signature_sha256'] = original_hash; bind()
        fixture['expected_outputs'][-1]['values'][-1] += .01; bind()
        check('runner-expected-values-differ', error='Runner expected outputs differ')
        fixture = copy.deepcopy(original_fixture); bind()
        # Appending bytes preserves NPZ values: a rejection must come from its
        # stored checksum binding rather than changed numerical reference data.
        with (temporary / 'reference-exact.npz').open('ab') as stream:
            stream.write(b'checker-test-only stale reference artifact')
        check('stale-reference-file-hash', error='Reference artifact hash')
        shutil.copyfile(source / 'reference-exact.npz', temporary / 'reference-exact.npz')
        with np.load(temporary / 'reference-approximate.npz', allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays['keys'] = np.concatenate((arrays['keys'], arrays['keys'][-1:]), axis=0)
        np.savez(temporary / 'reference-approximate.npz', **arrays)
        fixture['model']['reference_approximate_sha256'] = digest(temporary / 'reference-approximate.npz'); bind()
        check('reference-extra-layer-with-rebound-checksum', error='approximate keys must be finite')
        shutil.copyfile(source / 'reference-approximate.npz', temporary / 'reference-approximate.npz')
        fixture = copy.deepcopy(original_fixture); bind()
        check('positive-after-all-restores', passed=True)
        print(f'PASS {tests} checker-only cases; positive acceptance is simulated data, '
              'not complete-model CKKS numerical acceptance.', flush=True)


if __name__ == '__main__':
    main()
