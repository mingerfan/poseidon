"""Prepare the three missing native metadata cells, with no native execution.

The deterministic public vector is a microbenchmark fixture, never a model
calibration or quality sample. The original 36-cell entry and records are kept.
"""
import argparse
from array import array
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from .r10_paper_metrics import write_json
from .r31_lattigo_binding import NativeBinding
from .r31_lattigo_profile import make_profile_plan
from .r31_lattigo_profile_reference import (
    compare_references, direct_profile_sum, replay_profile,
)
from .r31_lattigo_runtime_plan import content_filename, plan_inventory

SLOTS = 1 << 14
PREPARATION_SECONDS = 30
CASE_BYTES = 4 * 1024 ** 2
MODE = 'independent-native-three-cell-preparation'
MODELS = ('ResNet20', 'VGG16', 'AlexNet')


def _operand(level=5, scale=50, kind='ciphertext', components=2):
    return dict(kind=kind, level=level, scale_log2=scale, components=components)


def _cell_key(cell):
    return json.dumps({key:cell[key] for key in ('op', 'inputs', 'output', 'attrs')},
                      sort_keys=True, separators=(',', ':'))


EXPECTED_CELLS = (
    dict(op='mod_switch', inputs=[_operand()], output=_operand(1),
         attrs=dict(target_level=1)),
    dict(op='mul_cp', inputs=[_operand(), _operand(kind='plaintext', components=1)],
         output=_operand(scale=100), attrs={}),
    dict(op='rotate', inputs=[_operand()], output=_operand(), attrs={}),
)


def select_missing_cells(coverage):
    """Require the exact three full operand/output metadata identities."""
    if coverage.get('status') != 'metadata_only_cost_transfer_coverage':
        raise ValueError('The existing common-model metadata coverage record is required')
    expected = {_cell_key(cell):cell for cell in EXPECTED_CELLS}
    selected = {key:dict(cell=copy.deepcopy(cell), source_counts={}, source_plans={})
                for key,cell in expected.items()}
    seen_models = set()
    for model in coverage['rows']:
        name = model['model']
        if name not in MODELS or name in seen_models or model.get('all_metadata_covered') is not False:
            raise ValueError('Require the three distinct unresolved common-model rows')
        seen_models.add(name)
        seen_cells = set()
        for cell in model['uncovered_cells']:
            key = _cell_key(cell)
            if key not in expected or key in seen_cells:
                raise ValueError('Unexpected or duplicate missing metadata cell')
            seen_cells.add(key)
            representative = cell.get('representative_attrs') or {}
            required = ({'steps':1023} if cell['op'] == 'rotate' else expected[key]['attrs'])
            if json.dumps(representative, sort_keys=True) != json.dumps(required, sort_keys=True):
                raise ValueError('Representative target/rotation attributes differ')
            count = cell.get('source_count')
            if type(count) is not int or count <= 0:
                raise ValueError('Positive original metadata source counts required')
            selected[key]['source_counts'][name] = count
            selected[key]['source_plans'][name] = model['source_plan']
        required_ops = {'mod_switch'} if name == 'ResNet20' else {'mod_switch', 'mul_cp', 'rotate'}
        if {expected[key]['op'] for key in seen_cells} != required_ops:
            raise ValueError('The original per-model missing-cell coverage differs')
    if seen_models != set(MODELS) or any(not row['source_counts'] for row in selected.values()):
        raise ValueError('All three missing metadata identities must be present')
    return [selected[key] for key in sorted(selected)]


def make_subset_source(selected):
    """One source instruction per selected cell; original plans are unopened."""
    if {_cell_key(row['cell']) for row in selected} != {_cell_key(x) for x in EXPECTED_CELLS} or len(selected) != 3:
        raise ValueError('Use exactly the three accepted metadata cells')
    place = dict(kind='host', rank=0)
    def desc(name, metadata):
        return dict(id=name, context='qwen-bootstrap48', ntt=True,
                    place=dict(place), **copy.deepcopy(metadata))
    values = [desc('subset-x', _operand()),
              desc('subset-plain', _operand(kind='plaintext', components=1))]
    instructions = []
    for index,row in enumerate(selected):
        cell = row['cell']; name = 'subset-target-' + str(index)
        values.append(desc(name, cell['output']))
        inputs = ['subset-x', 'subset-plain'] if cell['op'] == 'mul_cp' else ['subset-x']
        attrs = {'steps':1023} if cell['op'] == 'rotate' else copy.deepcopy(cell['attrs'])
        instructions.append(dict(kind='compute', ordinal=index+1, op=cell['op'],
                                 inputs=inputs, output=name, attrs=attrs, place=dict(place)))
    return dict(target=dict(target_id='pt-agent-independent-three-cell-source',
                            fixture=True, source_model_execution=False),
                values=values, execution=instructions)


def deterministic_fixture(slots=SLOTS):
    if type(slots) is not int or slots <= 0:
        raise ValueError('Positive fixture size required')
    return array('d', ((index % 17 - 8) / 32 for index in range(slots)))


def validate_prepared(prepared):
    prepared = Path(prepared).resolve()
    record = json.loads((prepared/'result.json').read_text())
    if (record.get('status') != 'ok' or record.get('phase') != 'prepare'
            or record.get('client_cache_protocol', 0) < 9
            or 50 not in record.get('supported_precision_bits', [])
            or 15 not in record.get('supported_mod1_inv_degrees', [])
            or 2 not in record.get('supported_log_message_ratios', [])
            or 3 not in record.get('supported_bootstrap_normalizations_log2', [])
            or record.get('supports_input_ssa_check') is not True):
        raise ValueError('Existing prepared client must support the exact independent binding')
    binary = Path(record['ready_binary'])
    if not binary.is_file():
        raise ValueError('The original prepared client is unavailable')
    return dict(result=str(prepared/'result.json'), binary=str(binary),
                binary_bytes=binary.stat().st_size, client_cache_protocol=record['client_cache_protocol'],
                binary_executions=0, content_checks=0)


def execution_command(preparation, *, out, prepared, seconds=300):
    """The existing supervised execute phase reuses its prepared binary directly."""
    if type(seconds) is not int or not 1 <= seconds <= 900:
        raise ValueError('Finite future native allowance of1..900seconds required')
    return [sys.executable, '-B', '-m', 'pt_agent.experiments.r31_lattigo_runtime_plan',
            'execute', '--out', str(out), '--prepared', str(prepared),
            '--plan', preparation['output_plan'], '--bundle', preparation['output_bundle'],
            '--input', preparation['input'], '--expected', preparation['expected'],
            '--bootstrap-cache', '32', '--precision-bits', '50', '--mod1-inv-degree', '15',
            '--log-message-ratio', '2', '--bootstrap-normalization-log2', '3',
            '--check-input-ssa', '--trace-key-requests', '--seconds', str(seconds),
            '--native-address-space-gib', '6', '--native-rss-gib', '2',
            '--minimum-available-gib', '1.25', '--native-go-memory-limit', '1500MiB',
            '--case-gib', '1']


def prepare(args):
    began = time.perf_counter_ns()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', mode=MODE, error=None,
                  created=datetime.now(timezone.utc).isoformat(),
                  coverage=str(Path(args.coverage).resolve()),
                  native_executions=0, keygen_runs=0, encryption_runs=0,
                  bootstrap_runs=0, decryption_runs=0, model_forwards=0,
                  calibration_reads=0, heldout_reads=0, llm_calls=0, downloads=0,
                  installations=0, extra_integrity_checks=0,
                  security_acceptance=None, accuracy_acceptance=None,
                  formal_prediction=None, formal_matrix_complete=False,
                  fixture_scope='Deterministic public arithmetic microbenchmark; no model/calibration/quality samples')
    def check():
        if (time.perf_counter_ns()-began)/1e9 > PREPARATION_SECONDS:
            raise TimeoutError('Subset preparation allowance exhausted')
        if sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) > CASE_BYTES:
            raise RuntimeError('Subset preparation artifact allowance exhausted')
    try:
        coverage = json.loads(Path(args.coverage).read_text())
        selected = select_missing_cells(coverage)
        prepared = validate_prepared(args.prepared)
        binding = NativeBinding(50, mod1_inv_degree=15, log_message_ratio=2,
                                bootstrap_normalization_log2=3)
        source = make_subset_source(selected)
        plan, manifest, metadata = make_profile_plan(source, binding=binding)
        inventory = plan_inventory(plan, manifest, binding=binding)
        if (metadata['cell_count'], metadata['setup_and_aggregation_count'],
                len(plan['execution']), inventory['bootstrap_count'],
                inventory['encode_count']) != (3, 5, 8, 0, 1):
            raise ValueError('Expected3targets/5setup/8ops/0boot/1plaintext')
        if inventory['rotation_steps'] != [1023]:
            raise ValueError('Original representative rotation must be1023')
        packed = deterministic_fixture()
        constants = {row['content']:array('d', [row['value']])*SLOTS
                     for row in metadata['constant_vectors']}
        replay, summary = replay_profile(plan, constants, packed)
        direct = direct_profile_sum(metadata, packed, precision_bits=50)
        comparison = compare_references(direct, replay)
        if comparison['passed'] is not True or comparison['shape'] != [SLOTS]:
            raise ValueError('Full independent direct/interpreter reference comparison failed')
        check()
        bundle = out/'profile.bundle'; (bundle/'data').mkdir(parents=True)
        # The subset has its own locator; original36-cell artifacts stay intact.
        plan['plaintext_bundle']['id'] = 'r31-independent-three-missing-cells-precision50'
        manifest['bundle_id'] = plan['plaintext_bundle']['id']
        metadata.update(selected_source_cells=selected,
                        fixture_scope=result['fixture_scope'],
                        source_coverage=result['coverage'],
                        bootstrap_is_mathematical_identity=False,
                        bootstraps='No bootstrap targets or execution operations in this subset')
        write_json(out/'profile-plan.json', plan); write_json(bundle/'manifest.json', manifest)
        for content,vector in constants.items():
            raw = array('d', vector)
            if sys.byteorder != 'little': raw.byteswap()
            (bundle/'data'/content_filename(content)).write_bytes(raw.tobytes())
        raw = array('d', packed)
        if sys.byteorder != 'little': raw.byteswap()
        (out/'input.bin').write_bytes(raw.tobytes())
        write_json(out/'expected.json', dict(indices=list(range(SLOTS)), values=list(direct),
                   scope=result['fixture_scope'], math_comparison=comparison))
        write_json(out/'profile-metadata.json', metadata)
        write_json(out/'reference-summary.json', summary)
        write_json(out/'source-coverage.json', coverage)
        result.update(status='prepared', prepared_client=prepared,
                      output_plan=str(out/'profile-plan.json'), output_bundle=str(bundle),
                      input=str(out/'input.bin'), expected=str(out/'expected.json'),
                      metadata=str(out/'profile-metadata.json'), cell_count=3,
                      target_execution_count=3, setup_and_aggregation_count=5,
                      execution_count=8, reference_comparison=comparison,
                      reference_interpreter=summary, inventory=inventory,
                      native_binding=binding.metadata())
        result['future_execute_argv'] = execution_command(result, out=out.with_name(out.name+'-native'),
                                                         prepared=Path(args.prepared).resolve())
        result['future_execution_scope'] = 'Existing R31 execute outer supplies R10 supervision, physical memory/disk guards, unique eight-field registry and persisted completion. Its execute branch reads the prepared binary and does not build or download.'
        check()
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__+': '+str(exc))
    finally:
        result['recorded_elapsed_ns'] = time.perf_counter_ns()-began
        result['timing_scope'] = 'Pure preparation and two clear references only; native timings remain unmeasured. Final result persistence follows this clock.'
        write_json(out/'preparation.json', result)
        if sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) > CASE_BYTES:
            result.update(status='error', error='Final preparation JSON exceeded the artifact allowance')
            write_json(out/'preparation.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('coverage', 'prepared', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args)
    print(json.dumps(dict(status=result['status'], out=str(args.out), error=result['error'])))
    return 0 if result['status'] == 'prepared' else 1


if __name__ == '__main__':
    raise SystemExit(main())
