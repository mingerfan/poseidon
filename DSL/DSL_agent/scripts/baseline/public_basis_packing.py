"""Versioned scalar banks and reusable geometry, without a model-specific DSL."""
import itertools
import hashlib
import math
import struct
from benchmark_graph import validate, require, canonical
from public_geometry_basis import permutation_layers, complete_partial

VERSION = 'capacity-public-tensors-v11'
MAX_CONSTANTS = 256


def flatten(value):
    if type(value) is list:
        for child in value:
            yield from flatten(child)
    else:
        require(type(value) in (int, float), 'Public basis numeric coefficient')
        number = float(value)
        require(math.isfinite(number) and abs(number) <= 1024, 'Public basis coefficient magnitude')
        yield number


class Registry:
    def __init__(self, period):
        self.period = period
        self.values = {f'p{i}': value for i, value in enumerate((-2., -1., 0., .5, 1., 2.))}
        self.seen = {}
        self.networks = {}
        self.network_cache = {}
        self.bits = {}
        self.add([1.] + [0.] * (period - 1), 'mask0')

    def add(self, value, name=None):
        require(len(value) == self.period and all(math.isfinite(v) and abs(v) <= 1024 for v in value),
                'Public basis vector shape/value')
        # Preserve signed zero and every binary64 coefficient bit.
        key = struct.pack('<' + str(len(value)) + 'd', *value)
        if key in self.seen:
            return self.seen[key]
        require(len(self.values) < MAX_CONSTANTS, 'Capacity public tensor registry budget')
        name = name or 'gv' + str(len(self.values))
        require(name not in self.values, 'Public basis unique registry name')
        self.values[name] = value
        self.seen[key] = name
        return name

    def mask(self, positions):
        selected = set(positions)
        if len(selected) == self.period:
            return None
        return self.add([float(i in selected) for i in range(self.period)])

    def prefix(self, count):
        require(0 < count <= self.period, 'Public basis prefix length')
        return self.mask(range(count))

    def network(self, permutation):
        key = tuple(permutation)
        if all(i == value for i, value in enumerate(key)):
            return None
        if key in self.network_cache:
            return self.network_cache[key]
        rows = []
        for distance, swaps in permutation_layers(key):
            if distance not in self.bits:
                self.bits[distance] = self.add([float((i & distance) == 0) for i in range(self.period)])
            rows.append(dict(distance=distance, low_half=self.bits[distance], swap=self.mask(swaps)))
        name = 'net' + str(len(self.networks))
        self.networks[name] = rows
        self.network_cache[key] = name
        return name

    def partial(self, pairs):
        permutation, selected, before, after = complete_partial(pairs, self.period)
        return dict(input_shift=before, output_shift=after,
                    network=self.network(permutation), mask=self.mask(selected))


def packing(model, plan):
    from unified_capacity_spatial import geometry, OPS as SPATIAL_OPS, MAX_ROUTE_GROUPS
    from unified_capacity_routes import contributions, OPS as ROUTE_OPS
    check = validate(model)
    shapes = check['shapes']
    period = plan['input_slot_period']
    require(type(period) is int and 4 <= period <= 4096 and period & (period - 1) == 0,
            'Public basis registered period')
    registry = Registry(period)
    scalars = []
    segments = {}
    derivations = {}

    def segment(name, value, shape):
        flat = list(flatten(value))
        require(len(flat) == math.prod(shape), 'Public basis coefficient shape')
        segments[name] = dict(offset=len(scalars), elements=len(flat), shape=list(shape))
        scalars.extend(flat)

    for name in sorted(model['constants']):
        segment(name, model['constants'][name], shapes[name])
    for node in model['nodes']:
        if node['op'] == 'batch_norm':
            mean, variance, gamma, beta = [model['constants'][ref] for ref in node['inputs'][1:]]
            gains = [float(g) / math.sqrt(float(v) + node['attrs']['eps']) for v, g in zip(variance, gamma)]
            offsets = [float(b) - float(m) * g for m, b, g in zip(mean, beta, gains)]
            gain = '@' + node['id'] + '.gain'
            offset = '@' + node['id'] + '.offset'
            segment(gain, gains, [len(gains)])
            segment(offset, offsets, [len(offsets)])
            derivations[node['id']] = dict(kind='batch_norm_affine', gain=gain, offset=offset,
                                          original_refs=node['inputs'][1:], eps=node['attrs']['eps'])
    pages = []
    for index, start in enumerate(range(0, len(scalars), period)):
        chunk = scalars[start:start + period]
        pages.append(registry.add(chunk + [0.] * (period - len(chunk)), 'wb' + str(index)))
    prefix_counts = {1}
    for shape in shapes.values():
        count = math.prod(shape) % period
        if count:
            prefix_counts.add(count)
    prefixes = {str(n): registry.prefix(n) for n in sorted(prefix_counts)}
    gather_cache = {}
    gathers = {}
    geometries = {}
    geometry_cache = {}
    node_views = {}

    def gather(source_spatial, target_spatial, stride, starts):
        signature = tuple(source_spatial), tuple(target_spatial), tuple(stride), tuple(starts)
        if signature in gather_cache:
            return gather_cache[signature]
        require(len(source_spatial) in (1, 2), 'Public basis spatial rank')
        if len(source_spatial) == 1:
            positions = [starts[0] + i * stride[0] for i in range(target_spatial[0])]
            require(len(set(positions)) == len(positions) and max(positions) < period and min(positions) >= 0,
                    'Public basis gather anchors')
            stages = [registry.partial(list(enumerate(positions)))]
        else:
            height, width = source_spatial
            out_height, out_width = target_spatial
            sy, sx = stride
            start_y, start_x = starts
            first = [(y * width + x, (start_y + y * sy) * width + x)
                     for y in range(out_height) for x in range(width)]
            second = [(y * out_width + x, y * width + start_x + x * sx)
                      for y in range(out_height) for x in range(out_width)]
            require(len(first) <= period and len(second) <= period,
                    'Public basis intermediate spatial plane')
            stages = [registry.partial(first), registry.partial(second)]
        name = 'gather' + str(len(gathers))
        gathers[name] = dict(source=list(source_spatial), target=list(target_spatial),
                             stride=list(stride), starts=list(starts), stages=stages,
                             final_mask=registry.prefix(math.prod(target_spatial)))
        gather_cache[signature] = name
        return name

    for node in model['nodes']:
        op = node['op']
        refs = node['inputs']
        if op in SPATIAL_OPS:
            g = geometry(node, shapes)
            src, dst = g['source'][2:], g['target'][2:]
            require(math.prod(src) <= period and math.prod(dst) <= period and
                    period % math.prod(src) == 0 and period % math.prod(dst) == 0,
                    'Public basis requires aligned channel planes')
            channels = g['source'][1] // g['groups'] if g['conv'] else 1
            require(math.prod(g['target']) * math.prod(g['kernel']) * channels <= MAX_ROUTE_GROUPS * period,
                    'Capacity spatial map work budget')
            signature = (tuple(src), tuple(dst), g['kernel'], g['stride'], g['padding'], g['dilation'], g['conv'],
                         node['attrs'].get('count_include_pad'))
            if signature not in geometry_cache:
                kernel = []
                normalizers = [0] * math.prod(dst)
                for offset in itertools.product(*(range(n) for n in g['kernel'])):
                    valid = []
                    for index, location in enumerate(itertools.product(*(range(n) for n in dst))):
                        position = [o * s - p + k * d for o, s, p, k, d in
                                    zip(location, g['stride'], g['padding'], offset, g['dilation'])]
                        if all(0 <= value < size for value, size in zip(position, src)):
                            valid.append(index)
                            normalizers[index] += 1
                    delta = 0
                    for size, pad, value, dilation in zip(src, g['padding'], offset, g['dilation']):
                        delta = delta * size + value * dilation - pad
                    kernel.append(dict(offset=list(offset), rotation=delta % period, mask=registry.mask(valid)))
                entry = dict(source_plane=math.prod(src), output_plane=math.prod(dst),
                             extract_mask=registry.prefix(math.prod(src)),
                             gather=gather(src, dst, g['stride'], (0,) * g['rank']), kernel=kernel)
                if not g['conv']:
                    counts = [math.prod(g['kernel'])] * len(normalizers) if node['attrs']['count_include_pad'] else normalizers
                    require(all(n > 0 for n in counts), 'Capacity pool window has no valid elements')
                    entry['normalizer'] = registry.add([1. / n for n in counts] + [0.] * (period - len(counts)))
                name = 'geometry' + str(len(geometries))
                geometries[name] = entry
                geometry_cache[signature] = name
            node_views[node['id']] = dict(kind='channel_spatial_basis', geometry=geometry_cache[signature],
                                         weight=refs[1] if g['conv'] else None,
                                         bias=refs[2] if g['conv'] and len(refs) == 3 else None)
        elif op == 'slice' and len(shapes[refs[0]]) >= 3 and node['attrs']['axis'] % len(shapes[refs[0]]) in (len(shapes[refs[0]]) - 2, len(shapes[refs[0]]) - 1):
            source = shapes[refs[0]][-2:]
            target = shapes[node['outputs'][0]][-2:]
            axis = node['attrs']['axis'] % len(shapes[refs[0]]) - (len(shapes[refs[0]]) - 2)
            start, stop, step = slice(node['attrs']['start'], node['attrs']['stop'], node['attrs']['step']).indices(source[axis])
            require(step > 0, 'Public basis positive spatial slice')
            stride = [1, 1]
            starts = [0, 0]
            stride[axis] = step
            starts[axis] = start
            node_views[node['id']] = dict(kind='channel_slice', gather=gather(source, target, stride, starts))
        elif op in ('sum', 'mean'):
            source = shapes[refs[0]]
            axes = sorted({i % len(source) for i in node['attrs']['axes']})
            require(axes == list(range(axes[0], len(source))), 'Public basis contiguous suffix reduction')
            width = math.prod(source[axes[0]:])
            require(width <= period and period % width == 0, 'Public basis suffix row alignment')
            rows = math.prod(source) // width
            anchors = {}
            for row in range(rows):
                src = row * width
                anchors.setdefault((src // period, row // period), []).append((row % period, src % period))
            node_views[node['id']] = dict(kind='suffix_window', width=width, rows=rows,
                factor=1. / width if op == 'mean' else 1.,
                anchors=[dict(input_chunk=a, output_chunk=b, **registry.partial(pairs))
                         for (a, b), pairs in sorted(anchors.items())])
        elif op in ROUTE_OPS:
            groups = {}
            for operand, src, dst, factor in contributions(node, shapes):
                require(factor == 1., 'Public basis permutation coefficient')
                key = operand, src // period, dst // period
                groups.setdefault(key, []).append((dst % period, src % period))
            require(len(groups) <= MAX_ROUTE_GROUPS, 'Capacity spatial route budget')
            node_views[node['id']] = dict(kind='partial_permutations', routes=[
                dict(operand=a, input_chunk=b, output_chunk=c, **registry.partial(pairs))
                for (a, b, c), pairs in sorted(groups.items())])
        elif op == 'linear':
            width = shapes[refs[0]][-1]
            rows = model['constants'][refs[1]]
            views = []
            if width <= period and len(rows) <= MAX_CONSTANTS - len(registry.values):
                for row in rows:
                    flat = list(flatten(row))
                    views.append(registry.add(flat + [0.] * (period - len(flat))))
            node_views[node['id']] = dict(kind='original_linear_coefficients', weight=refs[1],
                                         bias=refs[2] if len(refs) == 3 else None, row_views=views)
        elif op == 'batch_norm':
            node_views[node['id']] = dict(derivations[node['id']], representation='channel_affine_scalars')
        else:
            node_views[node['id']] = dict(kind='original_operator', op=op)
    descriptor = dict(version=VERSION, bindings=[], scalar_bank=dict(dtype='binary64', pages=pages,
        elements=len(scalars), segments=segments, derivations=derivations),
        geometry_basis=dict(networks=registry.networks, gathers=gathers, spatial=geometries,
                            node_views=node_views, prefixes=prefixes),
        source_model_sha256=hashlib.sha256(canonical(model)).hexdigest())
    return dict(vectors=registry.values, bindings=[], excluded_scalars=set(model['constants']),
                linear_strategies={}, linear_tiles={}, tensor_strategies={}, stack_operands={},
                spatial_windows={}, spatial_streams={}, suffix_windows={}, descriptor=descriptor)
