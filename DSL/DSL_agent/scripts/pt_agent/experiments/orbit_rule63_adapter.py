"""Own Orbit same-program adapter. Official Orbit files remain read-only.

All serialized Earth levels here are physical remaining levels. Orbit JSON index
0 denotes physical1; qwen48 arrays use the physical level as their direct index.
The adapted optimization table removes that sentinel; final prediction always
uses the untouched qwen48 table at the first ciphertext input level.
"""
from collections import Counter
from contextlib import contextmanager
import importlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import time
from types import MethodType

POLICY_ID = 'orbit-rule63-qwen48-fixed-1-to-5-20261117'
GIB = 1024 ** 3
OPS = {'constant', 'add', 'mul', 'negate', 'rotate', 'rescale', 'modswitch', 'upscale', 'bootstrap'}
LEVEL_SCOPE = dict(raw_earth_levels='unplaced placeholders, discarded by official compile-mode parser',
    decoded_levels='physical remaining level; no bts_ub-minus-level mirroring',
    initial_level=8, initial_scale=43, bootstrap_input_level=1,
    bootstrap_output_level=5, bootstrap_scale=43,
    orbit_table_index='index0 -> physical1', qwen48_table_index='array[physical_level]',
    same_table_prediction='first ciphertext input physical level, including rescale/modswitch/boot')
TYPE = re.compile(r'!ckks\.poly<\s*(\d+)\s*\*\s*(\d+)\s*\*\s*(\d+)\s*>')
DENSE = re.compile(r'value\s*=\s*dense<(.*?)>\s*:\s*tensor<([0-9x]+)xf64>', re.S)
FLOAT = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_profile(profile):
    expected = dict(polynomialDegree=65536, bootstrapLevelLowerBound=1,
        bootstrapLevelUpperBound=5, levelLowerBound=1, levelUpperBound=8, rescalingFactor=43)
    for key, value in expected.items():
        if profile.get(key) != value:
            raise ValueError('Same-config Orbit profile mismatch: ' + key)
    if not isinstance(profile.get('latencyTable'), dict):
        raise ValueError('Missing original qwen48 latencyTable')
    return expected


def map_types(line):
    """CKKS polynomial components1 are plaintext; >=2 are ciphertext."""
    def convert(m):
        components, scale, level = map(int, m.groups())
        if components < 1:
            raise ValueError('Invalid polynomial components')
        return f'!earth.{"pl" if components == 1 else "ci"}<{scale} * {level}>'
    result = TYPE.sub(convert, line)
    if '!ckks.' in result or '!poly.' in result:
        raise ValueError('Unrecognized CKKS/poly type or operation')
    return result


def dense_payload(text, count):
    """Return IEEE f64 storage without rounding hex payloads through decimal."""
    text = text.strip()
    if text.startswith('"0x') and text.endswith('"'):
        payload = bytes.fromhex(text[3:-1])
        # MLIR hex splats store one element; external .cst stores all elements.
        if len(payload) == 8 and count > 1:
            payload *= count
        if len(payload) != count * 8:
            raise ValueError('Dense hex payload shape differs')
        return payload
    if text.startswith('['):
        tokens = FLOAT.findall(text)
        if len(tokens) != count:
            raise ValueError('Dense decimal payload shape differs')
        values = [float(v) for v in tokens]
    else:
        if not FLOAT.fullmatch(text):
            raise ValueError('Unsupported dense f64 spelling')
        values = [float(text)] * count
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Nonfinite dense constant')
    return struct.pack('<' + 'd' * count, *values)


def adapt_mlir(source, output, constants, ledger_path, *, active_slots=16384, checkpoint=None):
    """Stream unchanged SSA operations; each dense f64 becomes an external ID.

    This is representation conversion, with no source-integrity/hash/readback
    checks. Payload bytes are consumed once to write the original .cst format.
    """
    if active_slots != 16384:
        raise ValueError('Rule63 active slots must remain16384')
    paths = [Path(output), Path(constants), Path(ledger_path)]
    if any(p.exists() or p.is_symlink() for p in paths):
        raise ValueError('Adaptation outputs must be fresh')
    original, converted, records = Counter(), Counter(), []
    count = 0
    with Path(source).open() as src, paths[0].open('x') as dst, paths[1].open('xb') as data:
        data.write(struct.pack('<q', 0))
        for number, line in enumerate(src, 1):
            if checkpoint:
                checkpoint()
            op_match = re.search(r'"earth\.(\w+)"', line)
            if line.lstrip().startswith('%'):
                if not op_match or op_match[1] not in OPS:
                    raise ValueError(f'Unsupported raw arithmetic operation on line{number}')
                original[op_match[1]] += 1
            transformed = map_types(line)
            if op_match and op_match[1] == 'constant':
                match = DENSE.search(transformed)
                rms = re.search(r'rms_var\s*=\s*([^,]+?)\s*:\s*f64', transformed)
                ssa = re.match(r'\s*(%\w+)\s*=', transformed)
                if not match or not rms or not ssa:
                    raise ValueError('Raw constant needs a dense f64 payload, rms_var and SSA identity')
                shape = [int(v) for v in match[2].split('x')]
                elements = math.prod(shape)
                payload = dense_payload(match[1], elements)
                # Literal hex values are checked for numerical finiteness only.
                if any(not math.isfinite(v[0]) for v in struct.iter_unpack('<d', payload)):
                    raise ValueError('Nonfinite dense hex constant')
                offset = data.tell()
                data.write(struct.pack('<q', elements))
                data.write(payload)
                transformed = transformed[:match.start()] + f'value = {count} : i64' + transformed[match.end():]
                if not math.isfinite(float(rms[1])):
                    raise ValueError('Nonfinite rms_var')
                records.append(dict(id=count, ssa=ssa[1], source_line=number, shape=shape,
                    elements=elements, payload_bytes=len(payload), cst_record_offset=offset,
                    rms_var_literal=rms[1], storage='little-endian IEEE754-f64',
                    original_value_preserved=True))
                count += 1
            dst.write(transformed)
            if op_match:
                converted[op_match[1]] += 1
        data.seek(0)
        data.write(struct.pack('<q', count))
    if original != converted or count != original.get('constant', 0):
        raise ValueError('Representation operation/constant inventory differs')
    result = dict(policy_id=POLICY_ID, source=str(source), adapted=str(output),
        constants=str(constants), ops_original=dict(original), ops_adapted=dict(converted),
        constants_count=count, records=records, arithmetic_operations_preserved=True,
        hashes=0, crc_checks=0, readback_checks=0, active_slots=active_slots)
    _save(ledger_path, result)
    return result


def _price(profile, op, level):
    values = profile['latencyTable'].get('earth.' + op)
    if (not isinstance(values, list) or type(level) is not int or not 1 <= level < len(values)
            or type(values[level]) not in (int, float) or not math.isfinite(values[level]) or values[level] < 0):
        raise ValueError(f'Missing qwen48 price for {op} at physicallevel{level}')
    return values[level]


def adapted_cost_config(profile):
    """Official cost-array index0 is physical1; keep fixed boot price input1."""
    validate_profile(profile)
    result = {k: v for k, v in profile.items() if k != 'latencyTable'}
    result['poly_deg'] = profile['polynomialDegree']
    result['latencyTable'] = {k: list(v[1:9]) for k, v in profile['latencyTable'].items()}
    result['latencyTable']['earth.upscale_single'] = [
        _price(profile, 'mul_single', level) for level in range(1, 9)]
    result['latencyTable']['earth.bootstrap_single'] = [_price(profile, 'bootstrap_single', 1)] * 8
    return result


def make_params(profile, Params, *, threads=1):
    validate_profile(profile)
    p = Params(None, 'Orbit', 'compile')
    p.poly_deg, p.max_slot = 65536, 16384
    p.bts_ub, p.bts_lb, p.lvl_ub, p.lvl_lb = 5, 1, 8, 1
    p.Sf = p.Sw = p.Csw = 43
    p.mode, p.sysname = 'compile', 'Orbit'
    p.backend = profile.get('runtime', '')
    p.latency_table = adapted_cost_config(profile)['latencyTable']
    p.bpsdepth, p.threads, p.comp, p.part, p.reqbp = 15, threads, True, True, False
    p.netname, p.trunc_val = 'same-rule63-16k', 1
    p.dacapo_mlir_in = p.dacapo_mlir_out = False
    def check_resbts(this, il, isc, ol, osc):
        if this.check_res(il, isc, ol, osc):
            return True
        return (this.check_res(il, isc, 1, 43) and this.check_res(5, 43, ol, osc))
    p.check_resbts = MethodType(check_resbts, p)
    return p


def make_estimator(params, profile, LatencyEstimator):
    class FixedEstimator(LatencyEstimator):
        def res_cost(self, il, isc, ol, osc):
            if not params.check_res(il, isc, ol, osc):
                raise ValueError('Invalid level-scale reduction')
            r = max(0, math.ceil((isc - osc) / 43))
            total = 0
            if ol + r < il:
                total += _price(profile, 'modswitch_single', il)
                il = ol + r
            for i in range(r - 1, -1, -1):
                target = min(2 * params.Sw, osc + 43 * i)
                if isc < target + 43:
                    total += _price(profile, 'mul_single', il)
                    isc = target + 43
                total += _price(profile, 'rescale_single', il)
                il -= 1
                isc -= 43
            if isc < osc:
                total += _price(profile, 'mul_single', il)
                isc = osc
            if (il, isc) != (ol, osc):
                raise ValueError('Reduction endpoint mismatch')
            return total

        def resbts_cost(self, il, isc, ol, osc):
            if params.check_res(il, isc, ol, osc):
                return self.res_cost(il, isc, ol, osc)
            if not params.check_resbts(il, isc, ol, osc):
                raise ValueError('Fixed1-to-5 bootstrap cannot reach target')
            return (self.res_cost(il, isc, 1, 43) + _price(profile, 'bootstrap_single', 1)
                    + self.res_cost(5, 43, ol, osc))
    estimator = FixedEstimator(params)
    estimator.lin_op_lmaps['bootstrap_single'] = (0, round(_price(profile, 'bootstrap_single', 1)))
    # Native objective uses a conservative constant rescale coefficient.
    estimator.lin_op_lmaps['rescale_single'] = (0, round(max(_price(profile, 'rescale_single', l)
        for l in range(2, 9))))
    return estimator


def fixed_decode_edge(fdag, u, v, scl):
    """Official decode reduction sequence, with physical bootstrap always1->5."""
    il, isc = fdag.nodes[u]['level'], fdag.nodes[u]['scale']
    ol = fdag.nodes[v]['level']
    osc = fdag.nodes[v]['scale'] if scl is None else scl
    def reduce(vc, cl, cs, tl, ts):
        r = max(0, math.ceil((cs - ts) / 43))
        if tl + r < cl:
            vc = fdag.insert_lsm_op('modswitch', vc, v, cl - tl - r)
            cl = tl + r
        for i in range(r - 1, -1, -1):
            target = min(2 * fdag.params.Sw, ts + 43 * i)
            if cs < target + 43:
                vc = fdag.insert_lsm_op('upscale', vc, v, target + 43 - cs)
                cs = target + 43
            vc = fdag.insert_lsm_op('rescale', vc, v, None)
            cl, cs = cl - 1, cs - 43
        if cs < ts:
            vc = fdag.insert_lsm_op('upscale', vc, v, ts - cs)
            cs = ts
        if (cl, cs) != (tl, ts):
            raise ValueError('Decoded reduction endpoint mismatch')
        return vc
    if fdag.params.check_res(il, isc, ol, osc):
        reduce(u, il, isc, ol, osc)
    else:
        if not fdag.params.check_resbts(il, isc, ol, osc):
            raise ValueError('Decoded fixed1-to-5 transition infeasible')
        vc = reduce(u, il, isc, 1, 43)
        vc = fdag.insert_lsm_op('bootstrap', vc, v, 5)
        reduce(vc, 5, 43, ol, osc)


def add_fixed_boot_constraints(tdag, vp, original):
    original(tdag, vp)
    for v in tdag.nodes:
        if tdag.nodes[v]['op'] == 'constant':
            continue
        vp.model.addGenConstrIndicator(vp.var_use(v, 'b'), True,
            vp.var_lvl(v, 'out') <= 5, name=f'fixed_boot5_output_upper_{v}')


@contextmanager
def fixed_boot_hooks(ilp_module, decoder_module):
    """Only in-process import objects change, restored even after an error."""
    original_constraints = ilp_module.add_ilp_constraints
    original_edge = decoder_module._decode_edge
    def constraints(tdag, vp):
        return add_fixed_boot_constraints(tdag, vp, original_constraints)
    ilp_module.add_ilp_constraints = constraints
    decoder_module._decode_edge = fixed_decode_edge
    try:
        yield
    finally:
        ilp_module.add_ilp_constraints = original_constraints
        decoder_module._decode_edge = original_edge


def decoded_observation(dag, profile):
    """Serialized per-cipher operation prediction, unlike compressed ILP score."""
    validate_profile(profile)
    counts, by_op = Counter(), Counter()
    missing, boot_records = [], []
    for label in dag.nodes:
        node = dag.nodes[label]
        op, weight = node['op'], node['weight']
        if type(weight) is not int or weight < 1:
            raise ValueError('Invalid decoded ciphertext multiplicity')
        if op in ('input', 'constant'):
            continue
        counts[op] += weight
        preds = list(dag.predecessors(label))
        cipher_preds = [u for u in preds if dag.nodes[u]['op'] != 'constant']
        if not cipher_preds:
            missing.append(dict(node=str(label), op=op, reason='no ciphertext input'))
            continue
        level = dag.nodes[cipher_preds[0]]['level']
        if op == 'bootstrap':
            if (level, dag.nodes[cipher_preds[0]]['scale'], node['level'], node['scale']) != (1, 43, 5, 43):
                raise ValueError('Decoded bootstrap differs from physical1/43-to-5/43')
            boot_records.append(dict(node=str(label), weight=weight, input_level=level,
                output_level=node['level'], input_scale=43, output_scale=43))
        key = dict(rotate='rotate_single', negate='negate_single', rescale='rescale_single',
            modswitch='modswitch_single', bootstrap='bootstrap_single', upscale='mul_single').get(op)
        if op in ('add', 'mul'):
            descriptor = node['op_descr']
            if (descriptor.get('single'), descriptor.get('double')) not in ((1, 0), (0, 1)):
                raise ValueError('Decoded arithmetic not single-call after official desquash')
            key = op + ('_single' if descriptor['single'] else '_double')
        if key is None:
            if op == 'relinearize':
                continue
            missing.append(dict(node=str(label), op=op, level=level, reason='operation has no qwen48 mapping'))
            continue
        try:
            by_op[op] += _price(profile, key, level) * weight
        except ValueError as exc:
            missing.append(dict(node=str(label), op=op, level=level, reason=str(exc)))
    total = sum(by_op.values())
    without = total - by_op.get('upscale', 0)
    return dict(compiled_bootstraps=sum(r['weight'] for r in boot_records),
        bootstrap_nodes=len(boot_records), bootstrap_records=boot_records, execution_ops=dict(counts),
        pred_latency_s=None if missing else total / 1e6,
        pred_latency_without_upscale_s=None if missing else without / 1e6,
        known_cost_partial_s=total / 1e6, known_cost_without_upscale_partial_s=without / 1e6,
        missing=missing, by_op_us=dict(by_op), excluded_operations=['relinearize'],
        upscale_price='same physical input level mul_single, explicit task policy',
        scope='decoded serialized one-inference operator sum; excludes keygen, encoding, host overhead; no fullwall calibration',
        rotation_scope='one serialized rotation call, independent of NAF surrogate used by optimizer',
        policy_id=POLICY_ID)


def serialized_bootstrap_calls(path):
    calls, records = 0, []
    with Path(path).open() as stream:
        for lineno, line in enumerate(stream, 1):
            if '"earth.bootstrap"' not in line:
                continue
            types = re.findall(r'tensor<(\d+)x!earth\.ci<(\d+)\s*\*\s*(\d+)>>', line)
            if len(types) != 2:
                raise ValueError('Serialized bootstrap needs one ciphertext input and output')
            before, after = [tuple(map(int, t)) for t in types]
            if before != (1, 43, 1) or after != (1, 43, 5):
                raise ValueError('Serialized bootstrap is not one physical1-to-5 call')
            calls += 1
            records.append(dict(line=lineno, calls=1, input_level=1, output_level=5))
    return dict(calls=calls, records=records, source=str(path))


def hecate_depth_copy(source, target):
    """Separate backend candidate: Hecate depth=initial8-physical remaining.

    Official Orbit physical IR remains the counting/projection artifact. This
    copy makes no claim of an executed native Hecate parser/typecheck.
    """
    pattern = re.compile(r'!earth\.(ci|pl)<(\d+)\s*\*\s*(\d+)>')
    def convert(m):
        level = int(m[3])
        if not 1 <= level <= 8:
            raise ValueError('Physical output level outside original profile')
        return f'!earth.{m[1]}<{m[2]} * {8-level}>'
    with Path(source).open() as src, Path(target).open('x') as dst:
        for line in src:
            line = pattern.sub(convert, line)
            line = re.sub(r'targetLevel\s*=\s*5\s*:\s*i64', 'targetLevel = 3 : i64', line)
            if '"func.func"' in line:
                line = line.replace('function_type = ', 'init_level = 8 : i64, function_type = ', 1)
            dst.write(line)
    return dict(path=str(target), level_representation='Earth rescale depth = initial8 minus physical',
        bootstrap_target_depth=3, initial_level=8, native_parser_executed=False,
        native_typecheck_executed=False)


def solve_fixed_input(partition, manager, comp, estimator, params):
    # With an explicit input budget, official multi-partition code stores the
    # merged whole result through add_qbp_existing and returns None. Retrieve
    # that stored table, preserving its exact fixed-input optimization.
    solved = partition.solve_partition(comp, manager, {8: {43: 0}}, estimator, params)
    if solved is None and comp.name in manager.pdag_name_to_qbp:
        solved = (manager.get_qbp_assign(comp), manager.get_qbp_cost(comp.name))
    return solved


def _modules(orbit_root):
    root = Path(orbit_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    names = dict(Params='scripts.params.params', parser='scripts.tdag.mlir2tdag',
        tdag='scripts.tdag', estimator='scripts.latency_estimator.latency_estimator',
        partition='scripts.optimizer.orbit.iterative_partition', qbp='scripts.optimizer.orbit.qbp_manager',
        decoder='scripts.assignment.decode_assign', writer='scripts.tdag.tdag2mlir',
        ilp='scripts.optimizer.orbit.ilp_gurobi')
    result = {k: importlib.import_module(v) for k, v in names.items()}
    for module in result.values():
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError('Orbit import resolved outside explicit original checkout')
    return result


def compile_rule63(*, source_mlir, profile, work_dir, orbit_root, active_slots=16384,
                   threads=1, checkpoint=None):
    """One adaptation/official parse/compress/ILP/decode/serialize attempt.

    The caller owns the total7200/RSS14/AS24 guard, source provenance, working
    directory isolation, completion, registry and paper publication.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=False)
    clock, stamp = {}, time.perf_counter()
    def stage(name, action):
        if checkpoint:
            checkpoint()
        start = time.perf_counter()
        value = action()
        clock[name] = time.perf_counter() - start
        _save(work / 'adapter-stage-times.json', clock)
        if checkpoint:
            checkpoint()
        return value
    adapted, cst, ledger = work / 'adapted.mlir', work / 'constants.cst', work / 'constant-ledger.json'
    inventory = stage('representation', lambda: adapt_mlir(source_mlir, adapted, cst, ledger,
        active_slots=active_slots, checkpoint=checkpoint))
    m = stage('imports', lambda: _modules(orbit_root))
    params = make_params(profile, m['Params'].Params, threads=threads)
    estimator = stage('cost_model', lambda: make_estimator(params, profile, m['estimator'].LatencyEstimator))
    _save(work / 'adapted-cost-config.json', adapted_cost_config(profile))
    dag = stage('parse', lambda: m['parser'].build_from_mlir(str(adapted), params))
    observed = Counter(dag.nodes[n]['op'] for n in dag.nodes if dag.nodes[n]['op'] != 'input')
    if dict(observed) != inventory['ops_adapted']:
        raise ValueError('Official parsed operation inventory differs from converted source')
    consts = {r['id']: r for r in inventory['records']}
    for n in dag.nodes:
        d = dag.nodes[n]
        if d['weight'] != 1:
            raise ValueError('Raw Rule63 must retain individual ciphertext SSA nodes')
        if d['op'] == 'constant':
            c = consts.get(d['op_descr']['value'])
            if c is None or c['ssa'] != '%' + str(n):
                raise ValueError('Official constant identity differs from externalization ledger')
    if any(dag.nodes[n]['op'] in ('rescale', 'upscale', 'modswitch', 'bootstrap') for n in dag.nodes):
        raise ValueError('Rule63 input must be raw unplaced arithmetic, without prior placement')
    if len(dag.inputs) != 1 or len(dag.outputs) != 1:
        raise ValueError('Official Orbit requires single input and output')
    def compress():
        m['tdag'].addition_squash(dag)
        m['tdag'].addition_desquash_bypass(dag)
        return m['tdag'].auto_compress(dag, is_ignore_weight=True)
    comp, mapping = stage('compression', compress)
    with fixed_boot_hooks(m['ilp'], m['decoder']):
        manager = stage('qbp_init', lambda: m['qbp'].QBPManager(params, estimator))
        solved = stage('integer_optimization', lambda: solve_fixed_input(
            m['partition'], manager, comp, estimator, params))
        if not solved or solved[0] is None or solved[1] is None:
            raise ValueError('Official Orbit produced no feasible assignment')
        assignments, costs = solved
        options = [(c, il, isc, ol, osc) for (il, isc), outs in costs.items()
            for (ol, osc), c in outs.items() if (il, isc) == (8, 43)]
        if not options:
            raise ValueError('Official Orbit has no assignment at original inputlevel8/scale43')
        cost, il, isc, ol, osc = min(options)
        assign = assignments[(il, isc)][(ol, osc)]
        decoded = stage('official_decode', lambda: m['decoder'].decode_assign(assign, dag, mapping))
    if any((decoded.nodes[n]['level'], decoded.nodes[n]['scale']) != (8, 43) for n in decoded.inputs):
        raise ValueError('Decoded original input differs from physical8/scale43')
    output = work / 'optimized.mlir'
    stage('serialization', lambda: m['writer'].tdag_to_mlir(decoded, str(output)))
    observation = stage('table_projection', lambda: decoded_observation(decoded, profile))
    serialized = stage('serialized_bootstrap_inventory', lambda: serialized_bootstrap_calls(output))
    if observation['compiled_bootstraps'] != serialized['calls']:
        raise ValueError('Decoded weighted bootstrap count differs from serialized single-inference calls')
    depth_copy = stage('hecate_depth_representation', lambda: hecate_depth_copy(output, work / 'optimized-hecate-depth.mlir'))
    result = dict(status='ok', policy_id=POLICY_ID,
        compile_verified=True, parameter_contract_verified=True,
        bootstrap_weighted_matches_serialized=True, source_operator_constant_bijection_verified=True,
        official_decoded_graph_check_passed=True, backend_native_typecheck_executed=False,
        execution_steps=sum(observation['execution_ops'].values()),
        serialization_scope='official Orbit physical-level SSA, single inference; separate Earth-depth candidate unexecuted',
        hecate_depth_candidate=depth_copy,
        compiled_bootstraps=observation['compiled_bootstraps'], prediction=observation,
        serialized_bootstrap_inventory=serialized, exclusive_stage_s=clock,
        adapter_wall_s=time.perf_counter() - stamp,
        artifacts=dict(adapted_mlir=str(adapted), constants_cst=str(cst), constant_ledger=str(ledger),
            optimized_mlir=str(output), optimized_hecate_depth_mlir=str(work / 'optimized-hecate-depth.mlir'), cost_config=str(work / 'adapted-cost-config.json')),
        input_inventory=dict(ops=inventory['ops_original'], constants=inventory['constants_count']),
        optimizer=dict(official_original_functions=True, original_library_disk_changes=0,
            fixed_bootstrap_constraints='original constraints plus indicator(boot -> output_level<=5)',
            fixed_decoder='original reduction sequence, normalized1/43 then fixedboot5/43',
            input_budget=dict(level=8, scale=43), bootstrap_cost_input_level=1,
            compressed_nodes=len(comp.nodes), decoded_nodes=len(decoded.nodes),
            surrogate_selected_cost_us=cost, surrogate_scope='original Orbit linearized cost, rotation NAF weights; final decoded projection is authoritative'),
        level_scope=LEVEL_SCOPE, scientific_quality_reexecuted=False,
        operator_projection_fullwall_calibration=False)
    _save(work / 'adapter-result.json', result)
    return result
