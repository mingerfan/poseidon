"""Read native HEIR artifacts and inventory control flow and coefficient storage."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import re
import sys

from .. import paths
from .heir_cnn_preflight import finalize_size, native_artifact_metadata
from .r10_paper_metrics import write_json


def analyze(mlir, parameters, go_module):
    text = mlir.read_text()
    native = native_artifact_metadata(mlir)
    residual = native['parameters'][0]
    actual = json.loads(parameters.read_text())
    if (residual['Q'] != actual['residual']['Q'] or residual['P'] != actual['residual']['P']
            or residual['logN'] != actual['residual']['logN']):
        raise ValueError('Actual constructor parameter group differs from compiled native literals')
    n = residual['ring_degree']
    p_count = len(residual['P'])
    functions = []
    current = None
    for line_number, line in enumerate(text.splitlines(), 1):
        match = re.match(r'  func\.func (?:private )?@([^\s(]+)', line)
        if match:
            current = dict(name=match[1], first_line=line_number, bootstraps=[], loops_or_conditions=[], calls=[])
            functions.append(current)
        if current is None:
            continue
        if '= lattigo.ckks.bootstrap ' in line:
            current['bootstraps'].append(dict(line=line_number, operation=line.strip()))
        if re.search(r'\b(?:affine|scf)\.(?:for|while|parallel|if)\b', line):
            current['loops_or_conditions'].append(dict(line=line_number, operation=line.strip()))
        match = re.search(r'\b(?:call|func.call) @([^\s(]+)', line)
        if match:
            current['calls'].append(dict(line=line_number, callee=match[1]))
    function_map = {row['name']: row for row in functions}
    body = function_map['resnet20_silu__preprocessed']
    entry = function_map['resnet20_silu']
    if body['loops_or_conditions'] or body['calls']:
        raise ValueError('Bootstrap-containing body requires control-flow interpretation')
    if [row['callee'] for row in entry['calls']].count(body['name']) != 1 or entry['loops_or_conditions']:
        raise ValueError('Entry does not invoke the bootstrap-containing body exactly once')
    if sum(len(row['bootstraps']) for row in functions) != len(body['bootstraps']):
        raise ValueError('Additional bootstrap-containing functions require call-graph analysis')
    pre = text[text.index('  func.func @resnet20_silu__preprocessing'):text.index('  func.func @resnet20_silu__preprocessed')]
    layouts = []
    for line_number, line in enumerate(text.splitlines(), 1):
        match = re.search(r'(%\w+) = call @(_assign_layout_\w+)\(.*-> memref<([\dx]+)xf64>', line)
        if match:
            shape = [int(value) for value in match[3].split('x')]
            layouts.append(dict(result=match[1], callee=match[2], line=line_number,
                                shape=shape, float64_payload_bytes=math.prod(shape) * 8))
    plaintext_levels = Counter(map(int, re.findall(r'lattigo.ckks.new_plaintext %param <level = (\d+)>', pre)))
    plaintext_bytes = sum((level + 1) * count * n * 8 for level, count in plaintext_levels.items())
    transforms = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if 'lattigo.ckks.prepare_linear_transform ' not in line:
            continue
        match = re.search(r'diagonal_indices = \[([^]]*)\].*levelQ = (\d+).*logSlots = (\d+)', line)
        if match is None:
            raise ValueError('Unknown native linear-transform syntax')
        indices = [int(value.strip()) for value in match[1].split(',')]
        if len(set(indices)) != len(indices):
            raise ValueError('Repeated linear-transform diagonal index')
        level = int(match[2])
        transforms.append(dict(line=line_number, diagonal_count=len(indices), levelQ=level,
                               levelP=p_count - 1, logSlots=int(match[3]),
                               QP_coefficient_payload_bytes=len(indices) * (level + 1 + p_count) * n * 8))
    first_transform_line = min(row['line'] for row in transforms)
    raw_before_first_transform = sum(row['float64_payload_bytes'] for row in layouts if row['line'] < first_transform_line)
    galois_elements = list(map(int, re.findall(r'lattigo.rlwe.gen_galois_key [^\n]*galoisElement = (\d+)', text)))
    q_count = len(residual['Q'])
    rns_digits = math.ceil(q_count / p_count)
    one_key_bytes = 2 * rns_digits * (q_count + p_count) * n * 8
    utils_source = (go_module / 'heircnn_utils/heircnn_utils.go').read_text()
    go_source = (go_module / 'heircnn.go').read_text()
    dimensions = re.findall(r'LogDimensions(?:\s*=|:)\s*ring.Dimensions\{Rows:\s*(\d+),\s*Cols:\s*(\d+)\}', go_source + '\n' + utils_source)
    if not dimensions or set(dimensions) != {('0', '14')}:
        raise ValueError('Generated Go packing dimensions differ from the verified logical packing')
    if utils_source.count('LevelP: param.GetRLWEParameters().MaxLevelP(),') != len(transforms):
        raise ValueError('Generated Go linear-transform P allocation differs')
    lintrans_bytes = sum(row['QP_coefficient_payload_bytes'] for row in transforms)
    return dict(
        status='ok', source_mlir=str(mlir), source_parameters=str(parameters), source_go_module=str(go_module),
        scope='Static native artifact and unchanged Lattigo allocation formulas; no model or cryptographic execution',
        model='resnet20_silu', native_residual_parameters=residual, full_constructor_parameters=actual,
        packing=dict(ring_degree=n, physical_max_slots=n // 2, logical_slots=16384,
                     generated_go_log_dimensions=[0, 14], assignments=len(dimensions)),
        functions=functions,
        bootstrap_count=dict(static_operations=len(body['bootstraps']),
                             derived_per_successful_complete_entry=len(body['bootstraps']),
                             derivation='Single unconditional entry call to straight-line preprocessed body; no loops, conditions or inner calls in that body',
                             actual_runtime_attempts=0, actual_completed_runtime_bootstraps=None),
        raw_layouts=layouts, raw_layout_total_bytes=sum(row['float64_payload_bytes'] for row in layouts),
        raw_layout_bytes_allocated_before_first_transform=raw_before_first_transform,
        encoded_plaintexts=dict(count=sum(plaintext_levels.values()), levels=dict(plaintext_levels),
                                Q_coefficient_payload_bytes=plaintext_bytes, retained_by_returned_array=True),
        linear_transforms=transforms, linear_transform_QP_coefficient_payload_bytes=lintrans_bytes,
        encoded_constants_coefficient_payload_bytes=plaintext_bytes + lintrans_bytes,
        ordinary_galois_keys=dict(count=len(galois_elements), unique_elements=len(set(galois_elements)),
                                  elements=galois_elements, rns_decomposition_digits=rns_digits,
                                  coefficient_payload_bytes_per_uncompressed_key=one_key_bytes,
                                  total_coefficient_payload_bytes=len(galois_elements) * one_key_bytes),
        relinearization_key_coefficient_payload_bytes=one_key_bytes,
        bootstrap_evaluation_key_payload_bytes=None,
        DFT_tables=dict(constructed=False, coefficient_payload_bytes=None,
                        S2C=actual['bootstrap_public_parameters']['SlotsToCoeffsParameters'],
                        C2S=actual['bootstrap_public_parameters']['CoeffsToSlotsParameters']),
        payload_scope='Coefficient-array payload; excludes object headers, samplers, encoder/evaluator buffers, DFT tables and bootstrap keys; this is not measured peak RSS',
        memory_basis=dict(plaintext='(levelQ+1)*N*8 per Plaintext',
                          linear_transform='diagonal_count*(levelQ+1+P_count)*N*8; generated Go sets LevelP=MaxLevelP',
                          evaluation_key='2*ceil(Q_count/P_count)*(Q_count+P_count)*N*8, uncompressed default, no base-two decomposition',
                          source_files=['Lattigo6.1.0 circuits/common/lintrans/lintrans.go:148-177',
                                        'Lattigo6.1.0 core/rlwe/keygenerator.go:239-244',
                                        'Generated heircnn_utils.go linear-transform literals']),
        native_eager_runtime_fits_16GiB=False,
        fit_reason='Returned encoded-constant arrays alone require more than 16GiB in coefficient payload',
        main_secret_generation='residual dense ternary P=2/3; same-N bootstrap extends this original secret',
        configured_bootstrap_Xs=actual['bootstrap']['Xs'],
        ephemeral_secret_weight=actual['ephemeral_secret_weight'],
        sparse_encapsulation_security='unresolved: this Lattigo6.1 cached source uses kgenDense for both encapsulation evaluation keys; do not infer exposure modulus from sparse secret storage alone',
        security_validated=False, keygen_runs=0, DFT_table_constructions=0, encryptions=0,
        real_ckks_attempts=0, model_forwards=0, llm_calls=0,
        compiled_bootstraps=None, predicted_latency=None, CKKS_latency=None, accuracy_acceptance=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--go-module', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.absolute()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(round='HEIR-CNN-INVENTORY', model='resnet20_silu', mode='native-artifact-memory-inventory', status='running', error=None)
    try:
        record.update(analyze(args.input.absolute(), args.parameters.absolute(), args.go_module.absolute()))
    except BaseException as error:
        record.update(status='error', error=type(error).__name__ + ': ' + str(error))
    record['ts'] = datetime.now(timezone.utc).isoformat()
    finalize_size(out, record)
    entry = dict(run_id=out.name, round=record['round'], model=record['model'], mode=record['mode'],
                 git_rev='working-tree', metrics={key: record.get(key) for key in
                 ('status', 'native_eager_runtime_fits_16GiB', 'compiled_bootstraps', 'predicted_latency', 'CKKS_latency', 'accuracy_acceptance')},
                 artifacts_dir=str(out), ts=record['ts'])
    with (paths.RESULTS_DIR / 'registry.jsonl').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
            raise ValueError('Existing result identity')
        stream.seek(0, 2)
        stream.write(json.dumps(entry) + '\n')
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)
    print(json.dumps({key: record.get(key) for key in ('status', 'error', 'raw_layout_total_bytes', 'encoded_constants_coefficient_payload_bytes', 'native_eager_runtime_fits_16GiB')}))
    return 0 if record['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
