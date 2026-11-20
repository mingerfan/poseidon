"""Fresh parent qualification and inference on the existing isolated CPU SDK.

No Provider, compiler, changed CKKS parameters, intermediate decrypt or new SDK.
The application owns the shared encrypted lease and artifact monitor. This
adapter verifies inputs and package bytes before and after the original runner.
"""
import copy
import json
from pathlib import Path
import subprocess
import tempfile

import cpu_composition_package as package

need, canonical, digest, sha, loads = (package.need, package.canonical, package.digest, package.sha, package.loads)
QUALIFICATION = 'poseidon-fresh-CPU-Conv-parent-qualification-v2'


def execution_parameters(raw, layout, count):
    import numpy as np
    from poseidon_artifact_gate import PRIMES
    for key in ('degree', 'data_moduli', 'special_moduli', 'input_slot_period', 'decoded_slots',
                'input_ciphertexts', 'output_ciphertexts'):
        need(type(raw[key]) is int, 'strict CPU integer field')
    need(all(type(value) is int for key in ('q', 'p') for value in raw[key]), 'strict CPU moduli')
    need(raw['backend'] == 'poseidon_cpu' and raw['encrypted_execution'] is True and raw['bootstrap'] is False and
         raw['security'] == 'tc128' and raw['degree'] == 32768 and raw['data_moduli'] == 13 and
         raw['special_moduli'] == 1 and raw['q'] == list(PRIMES[:13]) and raw['p'] == list(PRIMES[13:]) and
         raw['executor_secret_key'] is False, 'original CPU parameters and executor boundary')
    period = layout['input_slot_period']
    need(raw['format'] == 'poseidon-agent-execution-v2' and raw['execution_abi'] == layout['execution_abi'] and
         raw['input_slot_period'] == raw['decoded_slots'] == period and raw['input_ciphertexts'] == 2 and
         raw['output_ciphertexts'] == 1, 'composition execution ABI')
    slots = np.asarray(raw['outputs'], dtype=np.float64)
    need(slots.shape == (count, 1, period) and np.isfinite(slots).all(), 'actual decoded result shape')
    need(all(type(x) in (int, float) for row in raw['outputs'] for vector in row for x in vector),
         'numeric CPU outputs')
    return slots


def comparison(actual, expected, atol, rtol):
    import numpy as np
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    need(actual.shape == expected.shape and actual.size > 0 and np.isfinite(actual).all() and
         np.isfinite(expected).all(), 'finite qualification arrays')
    errors = np.abs(actual - expected)
    tolerance = atol + rtol * np.abs(expected)
    return dict(compared_values=int(errors.size), failures=int(np.count_nonzero(errors > tolerance)),
        max_absolute_error=float(errors.max()), max_tolerance_ratio=float((errors / tolerance).max()),
        atol=atol, rtol=rtol)


def execute_prepared(folder, samples, *, evidence, checkpoint, artifact_observer=None):
    import os, sys
    import numpy as np
    from hecate_python_env import WORK, VENV
    from program_execution import normalize_samples, _pack_logical_samples, named_outputs
    from poseidon_validation import input_payload
    from poseidon_candidate_sandbox import command
    from poseidon_sandbox_policy import limits
    from native_execution_slots import native_slot
    from poseidon_runtime_paths import checked_sdk, SPEC
    need(os.environ.get('IN_NIX_SHELL') == 'pure' and Path(sys.prefix) == VENV, 'pinned composition worker')
    manifest, request = package.check_bundle(folder, checkpoint=checkpoint)
    folder, evidence = Path(folder), Path(evidence)
    model = loads(package.regular(folder / 'model.json'))
    logical_request = dict(model=model, layout=request['layout'])
    logical = normalize_samples(logical_request, samples)
    sdk = checked_sdk()
    need(sha(package.regular(SPEC.resolve(strict=True))) == request['spec_raw_sha256'], 'current SDK operator spec')
    need(evidence.is_absolute() and evidence.is_relative_to(WORK / 'results') and not evidence.exists(),
         'fresh monitored native evidence')
    evidence.mkdir(mode=0o700)
    if artifact_observer is not None:
        artifact_observer(evidence)
    packed = _pack_logical_samples(logical_request, logical)
    package.write(evidence / 'inputs.json', canonical(input_payload(packed, request['layout'])))
    package.write(evidence / 'payload.json', canonical(dict(format='poseidon-composition-native-input-v1',
        parent_request_binding=request['binding'], package_binding=manifest['binding'], sdk_binding=sdk['binding'])))
    generated = {name: sha(package.regular(evidence / name)) for name in ('inputs.json', 'payload.json')}
    argv = command(evidence / 'payload.json', evidence,
        ['/poseidon-agent-cpu', '/composition-plan.json', '/operator-spec.json', '/composition-bundle',
         '/out/inputs.json', '/out/execution.json'])
    where = argv.index('--chdir')
    argv[where:where] = ['--ro-bind', str(folder / 'plan.v2.json'), '/composition-plan.json',
                         '--ro-bind', str(folder / 'bundle'), '/composition-bundle']
    package.write(evidence / 'native-command.json', canonical(dict(argv=argv, original_sandbox=True,
        additional_mounts='two readonly data paths', sdk_binding=sdk['binding'], max_invocations=1, retries=0)))
    checkpoint()
    try:
        with native_slot(WORK / 'cache/agent-native-slots'):
            with (evidence / 'native.log').open('xb') as log:
                result = subprocess.run(['/usr/bin/timeout', '-k', '3s', '150', *argv],
                    cwd=evidence, env={'PATH': '/usr/bin:/bin', 'POSEIDON_PROBE_SECRET': 'must-not-survive'},
                    stdout=log, stderr=subprocess.STDOUT, close_fds=True, timeout=160, preexec_fn=limits)
        need(result.returncode == 0, 'original CPU composition execution failed')
    finally:
        after = {name: sha(package.regular(evidence / name)) for name in generated}
        package.write(evidence / 'generated-input-guard.json', canonical(package.seal(dict(
            format='poseidon-composition-generated-input-guard-v1', before=generated, after=after,
            unchanged=after == generated))))
        need(after == generated, 'generated composition inputs changed')
        final, final_request = package.check_bundle(folder, checkpoint=checkpoint)
        need(final == manifest and final_request == request and checked_sdk() == sdk,
             'composition package or SDK changed during execution')
    raw = loads(package.regular(evidence / 'execution.json'))
    layout = request['layout']; period = layout['input_slot_period']
    slots = execution_parameters(raw, layout, len(logical))
    flat = np.asarray([[row[c, s] for c, s in layout['output_selectors']] for row in slots])
    outputs = named_outputs(logical_request, flat, len(logical))
    report = package.seal(dict(format='poseidon-composition-local-execution-v1', status='executed',
        bundle_binding=manifest['binding'], request_binding=request['binding'], authority=request['authority'],
        outputs=outputs, sdk_binding=sdk['binding'], samples=len(logical), encrypted_execution=True,
        executor_secret_key=False, secret_key_owner='original CPU acceptance owner process',
        final_only_decryption=True, single_key_epoch_within_this_invocation=True,
        numerically_validated_on_supplied_inputs=False, agent_calls=0, compiler_calls=0,
        execution_raw_sha256=sha(package.regular(evidence / 'execution.json')),
        generated_input_guard_sha256=sha(package.regular(evidence / 'generated-input-guard.json')),
        native_peak_CPU_or_RSS_measured=False))
    package.write(evidence / 'report.json', canonical(report))
    return report


def reference_material(folder, request):
    from cpu_composition_verified import references
    folder = Path(folder)
    model = loads(package.regular(folder / 'model.json'))
    children = [loads(package.regular(folder / 'children' / str(i) / 'model.json'))
                for i in range(len(request['children']))]
    need(package.digest(model) == request['model_sha256'] and
         [package.digest(child) for child in children] == [row['model_sha256'] for row in request['children']],
         'qualification reference model/order binding')
    return model, references(request, model, children)


def qualify(prepared, *, evidence, checkpoint, artifact_observer=None):
    import numpy as np
    manifest, request = package.check_bundle(prepared, checkpoint=checkpoint)
    model, verified = reference_material(prepared, request)
    report = execute_prepared(prepared, verified['probes'], evidence=evidence, checkpoint=checkpoint,
                              artifact_observer=artifact_observer)
    actual = np.asarray([[x for output in model['outputs'] for x in
        np.asarray(row[output['name']], dtype=np.float64).flat] for row in report['outputs']], dtype=np.float64)
    ckks = comparison(actual, verified['reference'], 1e-5, 1e-4)
    need(ckks['failures'] == 0, 'fresh parent CKKS qualification failed')
    receipt = package.seal(dict(format=QUALIFICATION, status='passed',
        parent_request_binding=request['binding'], prepared_package_binding=manifest['binding'],
        authority=request['authority'], profile=package.PROFILE, samples=4,
        math_comparison=verified['full_math_comparison'],
        child_leftfold_comparison=verified['child_leftfold_comparison'],
        reference_probes_f64_sha256=verified['probes_f64_sha256'],
        math_reference_sources=verified['math_sources'], reference_forward_counts=verified['forwards'],
        ckks_comparison=ckks,
        actual_execution_report_sha256=sha(package.regular(Path(evidence) / 'report.json')),
        actual_execution_raw_sha256=report['execution_raw_sha256'],
        generated_input_guard_sha256=report['generated_input_guard_sha256'],
        original_child_dependency_results=[row['original_current_dependency_result'] for row in request['children']],
        historical_child_manifests_unchanged=True, parent_agent_generated=False,
        final_only_decryption=True, agent_calls=0, compiler_calls=0))
    package.write(Path(evidence) / 'qualification.json', canonical(receipt))
    return receipt

def verify_qualification(folder, manifest, request):
    """No summary-only acceptance: bind raw execution and immutable input guard."""
    validation = manifest['validation']
    need(canonical(validation) == canonical(dict(level='numerical', encrypted_execution=True, numerically_validated=True)),
         'qualified parent grade')
    folder = Path(folder)
    receipt = package.bound(loads(package.regular(folder / 'qualification/qualification.json')))
    actual = package.bound(loads(package.regular(folder / 'qualification/report.json')))
    guard = package.bound(loads(package.regular(folder / 'qualification/generated-input-guard.json')))
    prepared = package.seal(dict({k: v for k, v in manifest.items() if k != 'binding'},
        files={k: v for k, v in manifest['files'].items() if not k.startswith('qualification/')},
        validation=dict(level='prepared', encrypted_execution=False, numerically_validated=False)))
    from poseidon_runtime_paths import checked_sdk
    sdk = checked_sdk()
    need(type(receipt['samples']) is int and type(actual['samples']) is int and
         all(type(item[key]) is int for item in (receipt, actual) for key in ('agent_calls', 'compiler_calls')),
         'strict qualification counts')
    need(receipt['format'] == QUALIFICATION and receipt['status'] == 'passed' and receipt['samples'] == 4 and
         receipt['parent_request_binding'] == request['binding'] and receipt['profile'] == package.PROFILE and
         receipt['prepared_package_binding'] == actual['bundle_binding'] == prepared['binding'] and
         actual['sdk_binding'] == sdk['binding'] and
         receipt['authority'] == actual['authority'] == request['authority'] and
         actual['request_binding'] == request['binding'] and actual['samples'] == 4 and
         actual['encrypted_execution'] is True and actual['agent_calls'] == actual['compiler_calls'] == 0 and
         receipt['actual_execution_report_sha256'] == sha(package.regular(folder / 'qualification/report.json')) and
         receipt['actual_execution_raw_sha256'] == actual['execution_raw_sha256'] ==
            sha(package.regular(folder / 'qualification/execution.json')) and
         receipt['generated_input_guard_sha256'] == sha(package.regular(folder / 'qualification/generated-input-guard.json')) and
         set(guard) == {'format', 'before', 'after', 'unchanged', 'binding'} and
         guard['format'] == 'poseidon-composition-generated-input-guard-v1' and
         guard['unchanged'] is True and set(guard['before']) == {'inputs.json', 'payload.json'} and
         guard['before'] == guard['after'] and
         all(sha(package.regular(folder / 'qualification' / name)) == expected
             for name, expected in guard['before'].items()), 'fresh parent execution provenance')
    # Recompute both comparisons using the unchanged original references and
    # stored raw outputs, not serialized success counters or trusted tolerance.
    import numpy as np
    model, verified = reference_material(folder, request)
    probes = verified['probes']
    from program_execution import _pack_logical_samples, named_outputs
    from poseidon_validation import input_payload
    logical_request = dict(model=model, layout=request['layout'])
    expected_inputs = canonical(input_payload(_pack_logical_samples(logical_request, probes), request['layout']))
    expected_payload = canonical(dict(format='poseidon-composition-native-input-v1',
        parent_request_binding=request['binding'], package_binding=prepared['binding'], sdk_binding=sdk['binding']))
    need(package.regular(folder / 'qualification/inputs.json') == expected_inputs and
         package.regular(folder / 'qualification/payload.json') == expected_payload and
         receipt['original_child_dependency_results'] ==
             [row['original_current_dependency_result'] for row in request['children']] and
         receipt['historical_child_manifests_unchanged'] is True and receipt['parent_agent_generated'] is False and
         receipt['agent_calls'] == receipt['compiler_calls'] == 0 and
         receipt['final_only_decryption'] is actual['final_only_decryption'] is True,
         'qualification inputs and original child history')
    def rows(values):
        return np.asarray([[x for output in model['outputs'] for x in np.asarray(row[output['name']]).flat]
                           for row in values], dtype=np.float64)
    reference = verified['reference']
    math_check = verified['full_math_comparison']
    child_check = verified['child_leftfold_comparison']
    raw = loads(package.regular(folder / 'qualification/execution.json'))
    layout = request['layout']
    slots = execution_parameters(raw, layout, 4)
    selected = np.asarray([[row[c, s] for c, s in layout['output_selectors']] for row in slots])
    need(actual['outputs'] == named_outputs(logical_request, selected, 4), 'qualification report raw outputs')
    ckks = comparison(selected, reference, 1e-5, 1e-4)
    need(canonical(receipt['child_leftfold_comparison']) == canonical(child_check) and
         child_check['failures'] == 0 and
         receipt['reference_probes_f64_sha256'] == verified['probes_f64_sha256'] and
         receipt['math_reference_sources'] == verified['math_sources'] and
         canonical(receipt['reference_forward_counts']) == canonical(verified['forwards']),
         'recomputed ordered child mathematical difference')
    need(canonical(receipt['math_comparison']) == canonical(math_check) and
         canonical(receipt['ckks_comparison']) == canonical(ckks) and
         math_check['failures'] == ckks['failures'] == 0, 'recomputed parent qualification')
    return receipt


def export_qualified(prepared, qualification, destination, *, checkpoint):
    prepared, qualification, destination = map(Path, (prepared, qualification, destination))
    manifest, request = package.check_bundle(prepared, checkpoint=checkpoint)
    need(not destination.exists(), 'fresh qualified parent destination')
    destination.mkdir(mode=0o700)
    files = dict(manifest['files'])
    total = 0
    for name, expected in files.items():
        raw = package.regular(prepared / name)
        need(sha(raw) == expected, 'prepared source changed during qualified export')
        total += len(raw)
        need(total <= package.MAX_BYTES, 'qualified parent storage bound')
        checkpoint(); package.write(destination / name, raw)
    for path in sorted(qualification.iterdir()):
        raw = package.regular(path)
        name = 'qualification/' + path.name
        total += len(raw)
        need(total <= package.MAX_BYTES, 'qualified evidence storage bound')
        package.write(destination / name, raw); files[name] = sha(raw); checkpoint()
    final = package.seal(dict({k: v for k, v in manifest.items() if k != 'binding'}, files=files,
        validation=dict(level='numerical', encrypted_execution=True, numerically_validated=True)))
    need(sum(path.stat().st_size for path in destination.rglob('*') if path.is_file()) <= package.MAX_BYTES,
         'qualified parent aggregate bound')
    package.write(destination / 'manifest.json', canonical(final))
    package.check_bundle(destination, level='numerical', checkpoint=checkpoint)
    return final
