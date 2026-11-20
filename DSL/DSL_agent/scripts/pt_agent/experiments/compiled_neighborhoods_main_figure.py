"""Compact editable paper scene from already audited RuntimePlan metadata.

Only the saved detailed-neighborhood JSON and the existing geometry helper are
read. No model, compiler, numerical payload or cryptographic client is loaded.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET


WIDTH = 504
HEIGHT = 356
MINIMUM_TEXT = 7.2
ROWS = (
    ('ResNet20', 'resnet20-fixed230', 'resnet20-fresh123'),
    ('VGG16', 'vgg16-fixed276', 'vgg16-fresh-no-tuner215'),
)
SHORT_OPS = {'rescale': 'Rescale', 'add_cp': 'AddCP', 'add_cc': 'AddCC',
             'mul_cp': 'MulCP', 'mul_cc': 'MulCC', 'relinearize': 'Relin',
             'boot': 'Boot', 'mod_switch': 'Drop', 'rotate': 'Rotate',
             'negate': 'Negate'}


def build(source, geometry):
    if source.get('font_family') != 'Arial':
        raise ValueError('Use the audited detailed Arial metadata source')
    indexed = {row['id']: row for row in source['rows']}
    if len(indexed) != len(source['rows']):
        raise ValueError('Duplicate detailed figure row')
    fig = geometry.Fig(WIDTH, HEIGHT, 'Actual compiled neighborhoods: own middle / first bootstrap')
    fig.text(10, 12, 'Compiled neighborhoods: own middle / first bootstrap', size=10.5, bold=True)
    fig.text(10, 26, 'ResNet20: each own middle boot. VGG16: each own first boot. Five actual operations per row.', size=7.5)
    selected = []
    ink, muted = '#17233A', '#526173'
    node_x, node_w, node_gap = 130, 65.2, 9
    for panel_index, (model, fixed_id, agent_id) in enumerate(ROWS):
        top = 36 + panel_index * 144
        fig.rect(8, top, 488, 136, fill='#F8FAFC', stroke='#CBD5E1', sw=.6, r=3)
        position_label = 'middle' if model == 'ResNet20' else 'first'
        fig.text(16, top + 11, '(' + chr(97 + panel_index) + ') ' + model + ': own ' + position_label + ' boot',
                 size=9, bold=True, color=ink)
        for row_index, identity in enumerate((fixed_id, agent_id)):
            row = indexed[identity]
            if row['model'] != model:
                raise ValueError('Model identity differs in detailed source')
            windows = [item for item in row['bootstrap_neighborhoods'] if item['label'] == position_label]
            if len(windows) != 1 or len(windows[0]['rows']) != 5:
                raise ValueError('Exactly one actual five-operation selected window is required')
            window = windows[0]
            ops = window['rows']
            sequence = row['bootstrap_count'] // 2 + 1 if position_label == 'middle' else 1
            if (window['bootstrap_sequence_number'] != sequence or
                    window['bootstrap_count'] != row['bootstrap_count'] or
                    ops[2]['op'] != 'boot' or ops[2]['index'] != window['execution_index'] or
                    [item['index'] for item in ops] != list(range(ops[0]['index'], ops[0]['index'] + 5))):
                raise ValueError('Detailed source does not describe its selected actual bootstrap')
            if identity.endswith('no-tuner215') and 'tuner disabled' not in row['variant_label']:
                raise ValueError('The VGG Agent row must retain its tuner-disabled identity')
            y = top + 25 + row_index * 57
            fixed = row_index == 0
            if fixed:
                label = ['Fixed Cheb63', str(row['bootstrap_count']) + ' total boots']
            elif model == 'VGG16':
                label = ['Agent; tuner disabled', str(row['bootstrap_count']) + ' total boots']
            else:
                label = ['Agent + tuner', str(row['bootstrap_count']) + ' total boots']
            fig.text(16, y + 10, label[0], size=8, bold=True, color=ink)
            fig.text(16, y + 22, label[1], size=7.5, color=muted)
            fig.text(16, y + 34, 'i' + str(ops[0]['index']) + '-' + str(ops[-1]['index']),
                     size=7.2, color=muted)
            position = {item['index']: n for n, item in enumerate(ops)}
            edge_groups = Counter()
            for edge in window['actual_SSA_edges']:
                left, right = position.get(edge['producer_index']), position.get(edge['consumer_index'])
                if left is None or right is None or left >= right:
                    raise ValueError('Local edge must connect two ordered in-window operations')
                producer, consumer = ops[left], ops[right]
                if producer['output'] != edge['value'] or edge['value'] not in consumer['inputs']:
                    raise ValueError('Local arrow is not an actual SSA producer/consumer dependency')
                edge_groups[(left, right, edge['value'])] += 1
            # Arrows describe SSA only. A missing dependency leaves the gap empty.
            for (left, right, _), multiplicity in edge_groups.items():
                start = (node_x + left * (node_w + node_gap) + node_w, y + 20)
                end = (node_x + right * (node_w + node_gap), y + 20)
                if right == left + 1:
                    points = [start, end]
                else:
                    lane = y - 4 - 3 * (right - left)
                    points = [start, (start[0] + 2, lane), (end[0] - 2, lane), end]
                fig.line(points, color='#526173', sw=.6, head=2.5)
            for n, item in enumerate(ops):
                x = node_x + n * (node_w + node_gap)
                boot = item['op'] == 'boot'
                fig.rect(x, y, node_w, 42, fill='#FFE6DC' if boot else '#E8F0FB',
                         stroke='#B94F32' if boot else '#7190B9', sw=.65, r=2)
                fig.text(x + node_w / 2, y + 10, SHORT_OPS[item['op']],
                         size=8, bold=True, anchor='c', color=ink)
                input_levels = list(dict.fromkeys(item['input_levels']))
                levels = ','.join(str(level) for level in input_levels)
                fig.text(x + node_w / 2, y + 22, 'L' + levels + ' -> ' + str(item['output_level']),
                         size=7.2, anchor='c', color=ink)
                inputs = list(dict.fromkeys(item['inputs']))
                identifiers = 'v' + inputs[0] + (',...' if len(inputs) > 1 else '') + '\u2192v' + item['output']
                if geometry.text_width(identifiers, 7.2) > node_w - 3:
                    identifiers = 'v' + inputs[0] + '\u2192v' + item['output']
                if geometry.text_width(identifiers, 7.2) > node_w - 3:
                    raise ValueError('Actual abbreviated SSA identifiers do not fit the compact node')
                fig.text(x + node_w / 2, y + 34, identifiers,
                         size=7.2, anchor='c', color=muted)
            selected.append(dict(model=model, id=identity, variant_label=row['variant_label'],
                source_plan=row['source_plan'], source_result=row['source_result'],
                total_compiled_bootstraps=row['bootstrap_count'], selected_window=window,
                displayed_edges=[dict(producer_index=ops[left]['index'], consumer_index=ops[right]['index'],
                                      value=value, operand_multiplicity=count)
                                 for (left, right, value), count in edge_groups.items()],
                omitted_external_edges=True))
    for y, text in (
        (327, 'Arrows show local SSA edges; dependencies outside each window are omitted.'),
        (338, 'v: first input -> output (other inputs omitted); i: execution index; L: level. Relin = relinearize.'),
        (349, 'Each row selects its own middle (ResNet20) or first (VGG16) boot; no semantic-layer alignment.'),
    ):
        fig.text(10, y, text, size=7.2, color=muted)
    text_items = [item for item in fig.items if item['k'] == 'text']
    overflow = [item for item in text_items if item['x'] < 0 or item['x'] + item['w'] > WIDTH
                or item['y'] - item['size'] * .65 < 0 or item['y'] + item['size'] * .35 > HEIGHT]
    if overflow or min(item['size'] for item in text_items) < MINIMUM_TEXT:
        raise ValueError('Compact scene text does not fit at the required native point size')
    scene = dict(width=WIDTH, height=HEIGHT, title=fig.title, items=fig.items)
    data = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Own middle ResNet20 and own first VGG16 bootstrap neighborhoods in saved original43/86 compiled RuntimePlans; compiled artifacts only.',
        selection_policy='For each ResNet20 row select its own floor(total_boots/2)+1 bootstrap; for each VGG16 row select its own first bootstrap. Retain two preceding and two following operations in each original execution order.',
        selection_policy_per_model=dict(ResNet20='middle', VGG16='first'),
        id_label_policy='First original input SSA identifier -> original output SSA identifier; additional inputs may be abbreviated. All original fields and operand multiplicities remain in selected_window.',
        semantic_layer_alignment=False, complete_four_system_matrix=False,
        source_detail_scope=source['scope'], rows=selected,
        abbreviations=SHORT_OPS, new_model_runs=0, new_compiler_runs=0, new_crypto_runs=0)
    review = dict(width_pt=WIDTH, height_pt=HEIGHT, width_inches=7,
        font_family='Arial', minimum_native_text_pt=min(item['size'] for item in text_items),
        minimum_text_pt_at_7inch_width=min(item['size'] for item in text_items),
        operator_and_system_label_pt=8, panels=2, actual_windows=4, actual_operations=20,
        actual_arrow_groups=sum(len(row['displayed_edges']) for row in selected),
        text_items=len(text_items), text_overflow=[],
        text_metrics='Existing figlib Arial-compatible advance widths; genuine Arial embedding is deferred to export_arial.',
        rendered_pdf_or_bitmap=False, old_detailed_figure_preserved=True)
    svg = fig.svg(scale=1).replace(
        'width="504" height="356"', 'width="504pt" height="356pt"', 1)
    ET.fromstring(svg)
    return scene, svg, data, review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--figlib', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location('compiled_scene_geometry', args.figlib)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    scene, svg, data, review = build(json.loads(args.source.read_text()), geometry)
    data['detailed_source'] = str(args.source.resolve())
    review['source_generator'] = str(Path(__file__).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / 'compiled-neighborhoods-main'
    for suffix, value in (('.scene.json', scene), ('.data.json', data), ('.readability.json', review)):
        stem.with_suffix(suffix).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    stem.with_suffix('.svg').write_text(svg + '\n')
    stem.with_suffix('.caption.txt').write_text(
        'Actual bootstrap neighborhoods from saved ResNet20 and VGG16 compiled RuntimePlans. '
        'Each ResNet20 row independently selects its own middle bootstrap (floor(total_boots/2)+1); '
        'each VGG16 row selects its own first bootstrap. Each window contains that operation and the '
        'two immediately preceding and two immediately following operations in its original execution order. '
        'Nodes show short operation names, input/output levels and the first original input SSA identifier '
        'followed by the original output SSA identifier; additional inputs are omitted where needed and retained in JSON. '
        'Only producer-consumer edges within that window are drawn; repeated uses of one SSA operand share an arrow. '
        'All dependencies crossing the window boundary are omitted. '
        'Fixed Cheb63 denotes an internal same-backend control; the VGG16 Agent row has tuning disabled. '
        'Total bootstrap counts are full-plan compiled counts. These independently selected windows do not imply '
        'semantic-layer alignment or a completed four-system matrix. Relin abbreviates relinearize. '
        'The first/middle/last detailed SVG and its complete metadata remain available as supplementary artifacts.\n')
    print(json.dumps(review, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
