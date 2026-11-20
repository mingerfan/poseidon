"""Static native component projection and actual compiled-neighborhood figure.

Only JSON/source metadata is read. This module neither executes a model or
cryptographic client nor opens a constant/weight/sample payload. The independent
profile is supplied explicitly; no complete-model native clock is an input.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def array_items(path, key):
    """Bounded decoder for one named top-level RuntimePlan array."""
    decoder = json.JSONDecoder()
    pattern = re.compile(r'"' + re.escape(key) + r'"\s*:\s*\[')
    with Path(path).open() as stream:
        buffer = ''
        while True:
            chunk = stream.read(65536)
            if not chunk:
                raise ValueError('Missing RuntimePlan array: ' + key)
            buffer += chunk
            found = pattern.search(buffer)
            if found:
                buffer = buffer[found.end():]
                break
            buffer = buffer[-128:]
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(','):
                buffer = buffer[1:].lstrip()
            if buffer.startswith(']'):
                return
            try:
                item, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = stream.read(65536)
                if not chunk:
                    raise ValueError('Truncated RuntimePlan array: ' + key)
                buffer += chunk
                continue
            yield item
            buffer = buffer[end:]


def lru(sequence, capacity, order=None):
    if type(capacity) is not int or capacity < 1:
        raise ValueError('Positive cache capacity required')
    order = list(order or [])
    hits = misses = 0
    for value in sequence:
        if value in order:
            hits += 1
            order.remove(value)
        else:
            misses += 1
            if len(order) >= capacity:
                order.pop(0)
        order.append(value)
    return order, dict(requests=hits + misses, hits=hits, misses=misses)


def scale(value):
    if value not in (43, 86):
        raise ValueError('Only original43/86 symbolic scale metadata is accepted')
    return {43: 50, 86: 100}[value]


def cell_key(cell):
    return json.dumps({key: cell[key] for key in ('op', 'inputs', 'output', 'attrs')},
                      sort_keys=True, separators=(',', ':'))


def check_sources(case, plan_path, record, quality_path, quality, stem):
    """Bind the actual compiled path and quality record to the same model."""
    factory='pt_agent.experiments.r13_common_models:'+stem
    if record.get('model') not in (stem,factory):
        raise ValueError('Compiled model identity differs from the selected row')
    compiled=record['phases']['compile']
    if (compiled.get('status')!='ok' or
            Path(compiled.get('runtime_plan','')).resolve()!=plan_path.resolve()):
        raise ValueError('RuntimePlan is not the saved successful compiled output')
    process=compiled['process']
    if (process.get('exit_code')!=0 or process.get('signal') is not None or process.get('timed_out') or
            process.get('signals_sent') or process.get('remaining_pids')):
        raise ValueError('Original native compilation was not cleanly completed')
    if 'qwen48' not in record.get('parameter_group',''):
        raise ValueError('Original compiler parameter group differs')
    if record.get('system')=='dacapo-depth-dp':
        source=record['source']
        program=quality['worker']['artifacts']['program']
        if (Path(source['source_result']).resolve()!=quality_path.resolve() or
                Path(source['program']).resolve()!=Path(program).resolve() or
                ast.dump(ast.parse(Path(compiled['source_program']).read_text()),include_attributes=False)!=
                ast.dump(ast.parse(Path(program).read_text()),include_attributes=False) or
                quality.get('model')!=factory or quality.get('status')!='ok' or
                quality.get('metrics',{}).get('heldout_acceptance') is not True):
            raise ValueError('Internal control quality/program provenance differs')
    else:
        worker=record['phases']['agent']['agent_result']
        if isinstance(worker,str):
            worker=read(worker)
        if (quality.get('status')!='quality_accepted' or
                quality.get('formal_frozen') is not True or
                quality.get('quality_acceptance_passed') is not True or
                quality.get('strict_math_passed') is not True or
                quality.get('model')!=stem or
                Path(quality['selected_program']).resolve()!=Path(worker['selected_program']).resolve()):
            raise ValueError('Fresh frozen quality/program provenance differs')
        # Fresh R30 records do not duplicate source_program in their compile
        # phase. Bind their actual preceding Emit invocation to the confirmed
        # program and this case's post-freeze output directory instead.
        emitted=[x for x in record.get('processes',[]) if x.get('phase')=='emit']
        if len(emitted)!=1:
            raise ValueError('Fresh compiled source has no unique Emit invocation')
        argv=emitted[0]['argv']
        def argument(name):
            if argv.count(name)!=1:raise ValueError('Missing unique Emit source argument')
            return Path(argv[argv.index(name)+1]).resolve()
        if (argument('--program')!=Path(quality['selected_program']).resolve() or
                argument('--post-dir')!=(case/'post-freeze').resolve()):
            raise ValueError('Fresh compiled Emit source differs from the frozen program')
    return dict(compiled_path_matches=True,clean_native_compile=True,
                same_model_quality_program_source=True,original_parameter_group=record['parameter_group'])


def inventory(path, lookup):
    """Stream full saved metadata, retaining only compact SSA descriptors."""
    values = {}
    contexts = set()
    for item in array_items(path, 'values'):
        name = item['id']
        if name in values:
            raise ValueError('Duplicate value descriptor')
        # Tuple retains original and explicit projected scale without the rest
        # of the large descriptor (place/context strings are not per-value copied).
        values[name] = (item['kind'], item['level'], item['scale_log2'], item['components'])
        scale(item['scale_log2'])
        contexts.add(item['context'])
    def metadata(name):
        kind, level, unit, components = values[name]
        return dict(kind=kind, level=level, scale_log2=scale(unit), components=components)
    encodes = {}
    for item in array_items(path, 'initialization'):
        if item['kind'] != 'encode' or item['payload']['kind'] != 'bundle':
            raise ValueError('Expected the compiled bundle-only initialization')
        encodes[item['output']] = item['payload']['content']
    cells = Counter()
    definitions = {}
    ops = Counter()
    rotations = []
    plain_levels = Counter()
    plain_hits = 0
    current_plain = None
    boot_indices = []
    rows = []
    # Rows contain tiny identity/arity metadata, not any numerical payload.
    for index, item in enumerate(array_items(path, 'execution')):
        if item['kind'] != 'compute':
            raise ValueError('Only compiled compute instructions are expected')
        attrs = {key: value for key, value in item.get('attrs', {}).items()
                 if key in ('target_level', 'target_scale_log2', 'target_components', 'implementation')}
        if 'target_scale_log2' in attrs:
            attrs['target_scale_log2'] = scale(attrs['target_scale_log2'])
        cell = dict(op=item['op'], inputs=[metadata(x) for x in item['inputs']],
                    output=metadata(item['output']), attrs=attrs)
        key = cell_key(cell)
        cells[key] += 1
        definitions[key] = cell
        ops[item['op']] += 1
        if item['op'] == 'boot':
            boot_indices.append(index)
        if item['op'] == 'rotate':
            step = item['attrs']['steps'] % 16384
            if step:
                rotations.append(pow(5, step, 131072))
        for name in item['inputs']:
            if name not in encodes:
                continue
            if name == current_plain:
                plain_hits += 1
            else:
                plain_levels[values[name][1]] += 1
                current_plain = name
        rows.append(dict(index=index, ordinal=item['ordinal'], op=item['op'],
                         inputs=item['inputs'], output=item['output'],
                         input_levels=[values[x][1] for x in item['inputs']],
                         output_level=values[item['output']][1],
                         output_scale_log2=values[item['output']][2]))
    if not boot_indices:
        raise ValueError('A bootstrapped full compiled plan is required')
    missing = [definitions[key] for key in cells if key not in lookup]
    if missing:
        raise ValueError('Independent profile does not cover cells: ' + str(missing))
    weighted = [dict(cell=definitions[key], source_count=count,
                     measured_core_ns=lookup[key]['core_ns'],
                     component_ns=count * lookup[key]['core_ns'])
                for key, count in cells.items()]
    _, rotate_cache = lru(rotations, 1)
    selected = []
    for label, position in [('first', 0), ('middle', len(boot_indices) // 2),
                            ('last', len(boot_indices) - 1)]:
        index = boot_indices[position]
        window = rows[max(0, index - 2):index + 3]
        outputs = {row['output']: row['index'] for row in window}
        edges = [dict(producer_index=outputs[name], consumer_index=row['index'], value=name)
                 for row in window for name in row['inputs'] if name in outputs]
        selected.append(dict(label=label, bootstrap_sequence_number=position + 1,
                             bootstrap_count=len(boot_indices), execution_index=index,
                             rows=window, actual_SSA_edges=edges))
    return dict(value_count=len(values), initialization_count=len(encodes),
                source_operations=len(rows), bootstrap_count=len(boot_indices),
                source_op_counts=dict(ops), unique_cells=len(cells), covered_cells=len(cells),
                contexts=sorted(contexts), weighted_cells=weighted,
                residual_cache1=dict(rotate_cache, unique_public_galois_elements=len(set(rotations))),
                plaintext_cache1=dict(misses_by_level=dict(plain_levels), hits=plain_hits),
                bootstrap_neighborhoods=selected)


CATALOG = (
    ('resnet20-fixed230', 'ResNet20', 'Fixed Cheb63 control', 230,
     'local-compile-depth-dp-no-debug-r29-20261116/dacapo-depth-dp',
     'work/emit/lowered._hecate_golden.runtime-plan.json',
     'r17-margin-heldout-20261115/resnet20_silu/result.json'),
    ('vgg16-fixed276', 'VGG16', 'Fixed Cheb63 control', 276,
     'local-compile-vgg16-depth-dp-r32-20261116/dacapo-depth-dp',
     'work/emit/lowered._hecate_golden.runtime-plan.json',
     'r17a-margin-heldout-20261115/vgg16_silu_avgpool/result.json'),
    ('alexnet-fixed204', 'AlexNet', 'Fixed Cheb63 control', 204,
     'local-compile-alexnet-depth-dp-r32-20261116/dacapo-depth-dp',
     'work/emit/lowered._hecate_golden.runtime-plan.json',
     'r17a-margin-heldout-20261115/alexnet_silu_avgpool/result.json'),
    ('resnet20-fresh123', 'ResNet20', 'Fresh Agent + tuner', 123,
     'fresh-resnet20-agent-e2e-r30-authorized-20261116',
     'post-freeze/emit/lowered._hecate_golden.runtime-plan.json',
     'fresh-resnet20-agent-e2e-r30-authorized-20261116/post-run-quality-audit.json'),
    ('vgg16-fresh-no-tuner215', 'VGG16', 'Fresh Agent; tuner disabled', 215,
     'fresh-vgg16-no-tuner-e2e-r30-20261116',
     'post-freeze/emit/lowered._hecate_golden.runtime-plan.json',
     'fresh-vgg16-no-tuner-e2e-r30-20261116/post-run-quality-audit.json'),
)


def project(source, measured, calibration):
    boot_cache = calibration['bootstrap_key_schedule']
    order, cold = lru(calibration['bootstrap_key_schedule']['per_boot_request_template'][:0], 32)
    # Independent constructor's actual request sequence is recorded separately
    # from the per-bootstrap sequence. Reconstruct it from the saved public trace.
    profile = Path(calibration['profile_case'])
    events = [json.loads(line) for line in (profile/'go-events.jsonl').read_text().splitlines()]
    requests = [row['galois_element'] for row in events
                if row.get('type') == 'galois_request' and row.get('cache') == 'bootstrap_lazy']
    constructor_count = boot_cache['constructor']['requests']
    order, cold = lru(requests[:constructor_count], 32)
    per_boot = []
    for _ in range(source['bootstrap_count']):
        order, observation = lru(boot_cache['per_boot_request_template'], 32, order)
        per_boot.append(observation)
    generated = cold['misses'] + sum(row['misses'] for row in per_boot)
    base = calibration['component_point_seconds']
    core = sum(row['component_ns'] for row in source['weighted_cells']) / 1e9
    encodes = source['plaintext_cache1']['misses_by_level']
    encode_units = calibration['plaintext_transfer']['per_level_measured_mean_seconds']
    encode_point = sum(count * encode_units[str(level)] for level, count in encodes.items())
    units = calibration['key_unit_seconds']
    components = dict(covered_operator_core_seconds=core,
                      bootstrap_galois_generation_seconds=generated*units['bootstrap'],
                      residual_galois_generation_seconds=source['residual_cache1']['misses']*units['residual'],
                      fixed_key_generation_seconds=base['fixed_key_generation_seconds'],
                      constructor_dft_seconds=base['constructor_dft_seconds'],
                      constructor_other_seconds=base['constructor_other_seconds'],
                      plaintext_read_encode_transfer_seconds=encode_point,
                      cold_parameters_input_encryption_final_seconds=base['cold_parameters_input_encryption_final_seconds'])
    return dict(component_point_seconds=components,
                engineering_component_point_seconds=sum(components.values()),
                bootstrap_cache32=dict(constructor=cold, per_boot=per_boot, total_generated=generated),
                formal_complete_prediction_seconds=None,
                recording_and_checkpoint_overhead_seconds=None,
                native_mathematical_acceptance=None, security_acceptance=None,
                per_cell_repeated_variance=None)


def render_svg(rows):
    """Editable Arial SVG; arrows are actual local SSA dependencies only."""
    by_id = {row['id']: row for row in rows}
    selected = [by_id[key] for key in ('resnet20-fixed230', 'resnet20-fresh123',
                                      'vgg16-fixed276', 'vgg16-fresh-no-tuner215')]
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1310" viewBox="0 0 1800 1310">',
           '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#475569"/></marker></defs>',
           '<style>text{font-family:Arial,sans-serif;fill:#17233a}.panel{fill:#f8fafc;stroke:#cbd5e1}.node{fill:#e8f0fb;stroke:#6a8bb8}.boot{fill:#ffe6dc;stroke:#c34c2e}.title{font-size:24px;font-weight:700}.label{font-size:16px}.small{font-size:12px}.tiny{font-size:10px}.edge{fill:none;stroke:#475569;stroke-width:1.6;marker-end:url(#arrow)}</style>',
           '<rect width="1800" height="1310" fill="white"/>',
           '<text x="38" y="40" font-size="28" font-weight="700">Actual compiled bootstrap neighborhoods</text>',
           '<text x="38" y="68" class="label">Saved RuntimePlans · first / middle / last bootstrap · local SSA dependencies · original symbolic scale43/86</text>']
    for panel, row in enumerate(selected):
        x = 38 + (panel % 2)*882
        y = 100 + (panel // 2)*548
        out.append(f'<g id="{row["id"]}"><rect class="panel" x="{x}" y="{y}" rx="10" width="850" height="520"/>')
        out.append(f'<text x="{x+18}" y="{y+32}" class="title">{html.escape(row["model"])} · {row["bootstrap_count"]} compiled bootstraps</text>')
        out.append(f'<text x="{x+18}" y="{y+58}" class="label">{html.escape(row["variant_label"])}</text>')
        for group, window in enumerate(row['bootstrap_neighborhoods']):
            sy = y + 96 + group*133
            label = f'{window["label"]}: bootstrap {window["bootstrap_sequence_number"]}/{window["bootstrap_count"]} · execution index {window["execution_index"]}'
            out.append(f'<g id="{row["id"]}-{window["label"]}"><text x="{x+18}" y="{sy}" class="small">{label}</text>')
            nodes = window['rows']
            positions = {item['index']: x+26+i*160 for i,item in enumerate(nodes)}
            for edge in window['actual_SSA_edges']:
                left = positions[edge['producer_index']]+142
                right = positions[edge['consumer_index']]
                top = sy+40
                if right-left < 22:
                    path=f'M {left},{top} L {right-3},{top}'
                else:
                    depth=18+5*(edge['consumer_index']-edge['producer_index'])
                    path=f'M {left},{top} C {left+12},{top-depth} {right-15},{top-depth} {right-3},{top}'
                out.append(f'<path class="edge" d="{path}"><title>SSA {html.escape(edge["value"])}: {edge["producer_index"]}→{edge["consumer_index"]}</title></path>')
            for item in nodes:
                nx=positions[item['index']]
                css='boot' if item['op']=='boot' else 'node'
                out.append(f'<g id="{row["id"]}-instruction-{item["index"]}-{window["label"]}"><rect class="{css}" x="{nx}" y="{sy+14}" rx="5" width="142" height="64"/>')
                out.append(f'<text x="{nx+8}" y="{sy+34}" font-size="15" font-weight="700">{html.escape(item["op"])}</text>')
                levels=','.join(map(str,item['input_levels']))+' → '+str(item['output_level'])
                out.append(f'<text x="{nx+8}" y="{sy+51}" class="small">level {levels}</text>')
                out.append(f'<text x="{nx+8}" y="{sy+67}" class="tiny">index {item["index"]}; ordinal {item["ordinal"]}</text></g>')
            out.append('</g>')
        out.append('</g>')
    out.extend(['<text x="38" y="1222" class="label">Neighborhoods follow each saved execution sequence; they are not semantic-layer alignments across models or planners.</text>',
                '<text x="38" y="1248" class="label">Only within-window producer→consumer SSA edges are drawn. Other operands and consumers remain outside the window.</text>',
                '<text x="38" y="1274" class="label">Fixed Cheb63 controls are internal same-backend comparisons. VGG Agent panel is a tuner-disabled ablation.</text>',
                '</svg>'])
    return '\n'.join(out)+'\n'


def execute(args):
    calibration = read(args.profile_prediction)
    measured = read(args.profile_audit)
    if measured['status'] != 'independent_saved_output_audit_passed':
        raise ValueError('Independent native profile audit is required')
    if Path(calibration['profile_case']).resolve()!=Path(measured['source']).resolve():
        raise ValueError('Timing and independent audit originate from different profiles')
    lookup = {cell_key(row['cell']): row for row in calibration['all39_measured_cells']}
    if len(lookup) != 39:
        raise ValueError('Exactly39 independently measured metadata cells required')
    rows = []
    for identity, model, variant, expected_boots, folder, plan_rel, quality_rel in CATALOG:
        case = args.results_root/folder
        record = read(case/'result.json')
        quality = read(args.results_root/quality_rel)
        if record['status'] != 'ok' or record['compiled_bootstraps'] != expected_boots:
            raise ValueError('Saved clean compilation identity/count mismatch: ' + identity)
        stem={'ResNet20':'resnet20_silu','VGG16':'vgg16_silu_avgpool','AlexNet':'alexnet_silu_avgpool'}[model]
        source_check=check_sources(case,case/plan_rel,record,args.results_root/quality_rel,quality,stem)
        source = inventory(case/plan_rel, lookup)
        if source['bootstrap_count'] != expected_boots:
            raise ValueError('Streamed bootstrap count mismatch: ' + identity)
        if source['source_op_counts']!=record['phases']['compile']['execution_ops']:
            raise ValueError('Streamed operations differ from the saved compile record')
        source.update(id=identity, model=model, variant_label=variant,
                      source_plan=str(case/plan_rel), source_result=str(case/'result.json'),
                      source_plan_bytes=(case/plan_rel).stat().st_size,
                      source_check=source_check,
                      parameter_group='Original qwen48 compiler metadata; explicit independent N16 native projection binding',
                      projected_native_variant=calibration['variant'],
                      symbolic_metadata_mapping='43→50;86→100, including target_scale_log2; arithmetic/levels/components/order unchanged',
                      quality_source=str(args.results_root/quality_rel),
                      quality_status=dict(heldout_acceptance=quality.get('metrics',{}).get('heldout_acceptance'),
                                          formal_frozen=quality.get('formal_frozen'),
                                          frozen_quality_accepted=quality.get('quality_acceptance_passed'),
                                          formal_accuracy_acceptance=None,
                                          scope='Controls use saved R17 heldout evidence only; frozen confirmation evidence is supplied for fresh Agent variants.'))
        source.update(project(source,measured,calibration))
        rows.append(source)
    result=dict(status='independent_native_component_matrix_projection',
                created=datetime.now(timezone.utc).isoformat(),
                scope='Source metadata, actual clean compiled neighborhoods and independent C2S52/P61 public39-cell timings only. No full native model timing is an input.',
                profile_prediction_source=str(args.profile_prediction),
                profile_audit_source=str(args.profile_audit),
                profile_variant=calibration['variant'],
                actual_constructor_metadata=calibration['actual_constructor_metadata'],
                profile_target_measurements_each=1,profile_target_repeated_variance=None,
                assumptions=[note.replace('real-plan182 Galois elements',
                    'each row\'s actual public Galois set (reported in residual_cache1)')
                    for note in calibration['assumptions']],rows=rows,
                source_model_runs=0,new_crypto_runs=0,new_API=0,
                full_native_model_clock_inputs=0,extra_integrity_checks=0,
                formal_complete_prediction_seconds=None,security_acceptance=None,
                figure_scope='Two models, two saved compiler plans each; first/middle/last compiled bootstrap neighborhoods, no inferred layer alignment. No four-system formal matrix claim.')
    args.out.mkdir(parents=True,exist_ok=True)
    write(args.out/'NATIVE_C2S52_COMPONENT_MATRIX_PROJECTION.json',result)
    (args.out/'COMPILED_BOOTSTRAP_NEIGHBORHOODS_TWO_MODELS.svg').write_text(render_svg(rows))
    write(args.out/'COMPILED_BOOTSTRAP_NEIGHBORHOODS_TWO_MODELS.json',dict(
        created=result['created'],scope=result['figure_scope'],font_family='Arial',
        rows=[{key:row[key] for key in ('id','model','variant_label','source_plan','source_result','bootstrap_count','source_operations','quality_status','bootstrap_neighborhoods')}
              for row in rows if row['model']!='AlexNet'],
        new_model_runs=0,new_crypto_runs=0,extra_integrity_checks=0))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-prediction',type=Path,required=True)
    parser.add_argument('--profile-audit',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--results-root',type=Path)
    args=parser.parse_args(argv)
    if args.results_root is None:
        from .. import paths
        args.results_root=paths.RESULTS_DIR
    result=execute(args)
    print(json.dumps([dict(id=row['id'],ops=row['source_operations'],boots=row['bootstrap_count'],
                           cells=row['unique_cells'],point_seconds=row['engineering_component_point_seconds']) for row in result['rows']]))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
