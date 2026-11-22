"""Explicit compact BERT scheduling with multi-prime logical rescales.

This is not the Cerium compiler: it preserves our polynomial BERT workload,
uses standard Poseidon arithmetic/Boot, and binds a prepared GPU dense Linear.
A 56-bit canonical scale and two ~28-bit physical primes per product mirror
one important Cerium scheduling choice. Exact profile/DFT/P policies can differ.
"""
from __future__ import annotations

from collections import Counter
import fnmatch
import math
from pathlib import Path
import numpy as np

try:
    from .backend import Graph, ManualLowering, json_bytes, digest
    from .packing_invariants import validate_partial_av_consumers
except ImportError:
    from backend import Graph, ManualLowering, json_bytes, digest
    from packing_invariants import validate_partial_av_consumers


AV_PARTIAL_ALGORITHM = 'PackedInput2Partial'
AV_PARTIAL_CALLEE = 'bert.compact_av_packed_partial'
AV_PARTIAL_OUTPUT_SEMANTICS = 'live_full_av_gap_B_current_plus_A_next_head_row'


def compact_av_partial_reference(scores, vectors, spec):
    """Defined ALL-slot partial function; standalone diagnostic, not admission.

    A/B partition key offsets by parity(d-j). Live=A+B, while each gap is
    B_here+A_next, with next advancing head then token (including final wrap).
    Scores retain all T offsets. This oracle models ideal real arithmetic only.
    """
    tokens, dim = spec['tokens'], spec['head_dim']
    if tokens != 2 * dim or spec['tile'] != 2 * tokens:
        raise ValueError('partial AV requires the cyclic two-head gap layout')
    scores, vectors = [np.asarray(a).reshape(tokens, 2, tokens) for a in (scores, vectors)]
    if not np.isfinite(scores).all() or not np.isfinite(vectors).all():
        raise ValueError('partial AV inputs must be finite in all slots')
    if np.any(vectors[:, :, dim:] != 0):
        raise ValueError('partial AV requires ideal zero value head gaps')
    a = np.zeros((tokens, 2, dim), dtype=np.result_type(scores, vectors, np.float64))
    b = np.zeros_like(a)
    features = np.arange(dim)
    for offset in range(tokens):
        term = scores[:, :, offset, None] * vectors[(np.arange(tokens) + offset) % tokens, :, :dim]
        even = (offset - features) % 2 == 0
        a[:, :, even] += term[:, :, even]
        b[:, :, ~even] += term[:, :, ~even]
    output = np.empty_like(vectors, dtype=a.dtype)
    output[:, :, :dim] = a + b
    output[:, :, dim:] = b + np.roll(a.reshape(2 * tokens, dim), -1, axis=0).reshape(a.shape)
    return output.reshape(-1)


class CompactGraph(Graph):
    """Keep public dense matrices out of the expanded primitive graph."""
    def __init__(self, slots, *, prepared_qk=False, prepared_av=False, qk_pack_inputs=False,
                 av_pack_inputs=False):
        super().__init__(slots)
        if qk_pack_inputs and not prepared_qk:
            raise ValueError('qk_pack_inputs requires prepared_qk')
        if av_pack_inputs and not prepared_av:
            raise ValueError('av_pack_inputs requires prepared_av')
        self.av_pack_inputs = bool(av_pack_inputs)
        self.matrices = {}
        self.use_prepared_qk = bool(prepared_qk)
        self.qk_pack_inputs = bool(qk_pack_inputs)
        self.qk_specs = {}
        self.use_prepared_av = bool(prepared_av)
        self.av_specs = {}

    def compact_linear(self, inputs, weight, bias, *, config, label):
        weight = np.ascontiguousarray(weight, dtype='<f4')
        bias = np.ascontiguousarray(bias, dtype='<f4')
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError('linear weight/bias dimensions do not match')
        if weight.shape[1] != len(inputs) * config.tile or weight.shape[0] % config.tile:
            raise ValueError('linear matrix is not feature-tiled')
        if not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise ValueError('linear weights must be finite')
        if label in self.matrices:
            raise ValueError('matrix labels must be unique: ' + label)
        self.matrices[label] = dict(weight=weight.copy(), bias=bias.copy(),
            tile=config.tile, baby_step=config.bsgs_baby_step)
        return [self.node('compact_linear', tuple(inputs), (label, tile))
                for tile in range(weight.shape[0] // config.tile)]

    def compact_qk(self, query, key, *, config, label=None):
        if not self.use_prepared_qk:
            return None
        if (config.slots != self.slots or config.tile != 2 * config.tokens or
                config.tokens != 2 * config.head_dim):
            raise ValueError('prepared QK requires the cyclic two-head gap layout')
        if query.backend is not self or key.backend is not self:
            raise ValueError('prepared QK inputs belong to another graph')
        if self.qk_pack_inputs:
            try:
                from .packing_invariants import ideal_head_gaps_are_zero
            except ImportError:
                from packing_invariants import ideal_head_gaps_are_zero
            if not all(ideal_head_gaps_are_zero(expr, config) for expr in (query, key)):
                raise ValueError('packed-input QK requires public proof of ideal zero head gaps for query and key')
        label = label or 'qk.' + str(len(self.qk_specs))
        if label in self.qk_specs:
            raise ValueError('prepared QK labels must be unique')
        self.qk_specs[label] = dict(tokens=config.tokens, tile=config.tile,
            head_dim=config.head_dim, baby=min(8, config.tokens))
        if self.qk_pack_inputs:
            self.qk_specs[label]['pack_inputs'] = True
        return self.node('compact_qk', (query, key), label)

    def compact_av(self, scores, value, *, config, label=None):
        if not self.use_prepared_av:
            return None
        if (config.slots != self.slots or config.tile != 2 * config.tokens or
                config.tokens != 2 * config.head_dim):
            raise ValueError('prepared AV requires the cyclic two-head gap layout')
        if scores.backend is not self or value.backend is not self:
            raise ValueError('prepared AV inputs belong to another graph')
        if self.av_pack_inputs:
            try:
                from .packing_invariants import ideal_head_gaps_are_zero
            except ImportError:
                from packing_invariants import ideal_head_gaps_are_zero
            if not ideal_head_gaps_are_zero(value, config):
                raise ValueError('partial AV requires public proof of ideal zero value head gaps')
        label = label or 'av.' + str(len(self.av_specs))
        if label in self.av_specs:
            raise ValueError('prepared AV labels must be unique')
        self.av_specs[label] = dict(tokens=config.tokens, tile=config.tile,
            head_dim=config.head_dim, baby=min(8, config.tokens))
        if self.av_pack_inputs:
            self.av_specs[label].update(algorithm=AV_PARTIAL_ALGORITHM,
                callee=AV_PARTIAL_CALLEE, output_semantics=AV_PARTIAL_OUTPUT_SEMANTICS,
                partial_output=True, client_value_gap_zero_required=True)
        return self.node('compact_av_partial' if self.av_pack_inputs else 'compact_av',
                         (scores, value), label)

    def validate_partial_av_consumers(self, outputs):
        validate_partial_av_consumers(self, outputs)

    def evaluate(self, outputs, inputs):
        outputs = list(outputs)
        self.validate_partial_av_consumers(outputs)
        values = {}
        for n in self.reachable(outputs):
            args = [values[a.index] for a in n.args]
            if n.op == 'input':
                value = np.asarray(inputs[n.data])
            elif n.op == 'plain':
                value = np.zeros(self.slots)
                payload = self.constants[n.data]
                value[:len(payload)] = payload
            elif n.op == 'boot':
                value = args[0]
            elif n.op == 'rotate':
                value = np.roll(args[0], -n.data)
            elif n.op == 'neg':
                value = -args[0]
            elif n.op == 'compact_qk':
                spec = self.qk_specs[n.data]
                tokens, dim = spec['tokens'], spec['head_dim']
                q, k = [a.reshape(tokens, 2, tokens)[:, :, :dim] for a in args]
                scores = np.einsum('thf,shf->ths', q, k) / math.sqrt(dim)
                offsets = (np.arange(tokens)[:, None] + np.arange(tokens)[None, :]) % tokens
                value = np.take_along_axis(scores, offsets[:, None, :], axis=2).reshape(-1)
            elif n.op == 'compact_av_partial':
                value = compact_av_partial_reference(*args, self.av_specs[n.data])
            elif n.op == 'compact_av':
                tokens = self.av_specs[n.data]['tokens']
                scores, vectors = [a.reshape(tokens, 2, tokens) for a in args]
                value = np.zeros_like(vectors, dtype=np.result_type(scores, vectors, np.float64))
                for offset in range(tokens):
                    value += scores[:, :, offset, None] * vectors[(np.arange(tokens) + offset) % tokens]
                value = value.reshape(-1)
            elif n.op == 'compact_linear':
                label, out_tile = n.data
                matrix = self.matrices[label]
                tile = matrix['tile']
                rows = self.slots // tile
                x = np.stack(args).reshape(len(args), rows, tile).transpose(1, 0, 2).reshape(rows, -1)
                first = out_tile * tile
                weight = matrix['weight'][first:first + tile].astype(np.float64)
                value = (x @ weight.T + matrix['bias'][first:first + tile]).reshape(-1)
            else:
                value = {'add': np.add, 'sub': np.subtract, 'mul': np.multiply}[n.op](*args)
            values[n.index] = value
        return [values[n.index] for n in outputs]


class MultiPrimeLowering(ManualLowering):
    """Canonical S[l-r] = 2*S[l] - sum(log2(q[l-r+1:l+1])).

    Both parity chains are seeded for r=2. Alignment is always an actual
    multiply-by-one plus rescale. No metadata-only scale or level edits occur.
    Prepared Linear owns its exact multiplication, sum, rescale and bias.
    """
    SELECTIVE = (
        '*.attention.qk.*',
        '*.attention.denominator.*',
        '*.attention_ln.variance.*',
        '*.attention_ln.output.*',
        '*.ffn_ln.variance.*',
        '*.ffn_ln.output.*',
    )

    def __init__(self, graph, context, profile, *, rescale_count=2,
                 working_scale_log2=None, strategy='selective', fuse=True,
                 initial_level=None, normalize_shared_rotations=False):
        if 2 * graph.slots != context['poly_degree']:
            raise ValueError('graph slots must match the physical CKKS ring')
        self.rescale_count = int(rescale_count)
        if self.rescale_count < 1:
            raise ValueError('rescale_count must be positive')
        super().__init__(graph, context, profile, strategy=strategy, fuse=fuse,
                         initial_level=initial_level)
        if 'q_moduli' in context:
            q = [int(x) for x in context['q_moduli']]
            if len(q) != len(self.q) or any(x <= 2 or (x - 1) % (2 * context['poly_degree']) for x in q):
                raise ValueError('invalid explicit Q chain')
            self.q = q
        self.anchor = int(profile['output_level']) - self.rescale_count
        self.initial = self.anchor if initial_level is None else int(initial_level)
        if not self.floor + self.rescale_count <= self.initial < len(self.q):
            raise ValueError('initial level has insufficient rescale budget')
        if self.anchor < self.floor + 2 * self.rescale_count:
            raise ValueError('insufficient native Boot output budget')
        working = (self.rescale_count * float(context['default_scale_log2'])
                   if working_scale_log2 is None else float(working_scale_log2))
        if not math.isfinite(working) or working <= 0:
            raise ValueError('invalid working scale')
        self.scales = [working] * self.rescale_count
        for level in range(self.rescale_count, len(self.q)):
            drop = self.drop_log2(level, self.rescale_count)
            self.scales.append((self.scales[level - self.rescale_count] + drop) / 2)
        self.prepared_calls = []
        self.prepared_qk_calls = []
        self.prepared_av_calls = []
        self.av_stats = Counter()
        self.qk_stats = Counter()
        self.linear_stats = Counter()
        self.normalize_shared_rotations = bool(normalize_shared_rotations)
        self.shared_rotation_frontiers = []

    def drop_log2(self, level, count):
        return sum(math.log2(q) for q in self.q[level - count + 1:level + 1])

    def normalize(self, x):
        if not x.pending:
            return x
        if x.id not in self.normalized:
            target = x.level - self.rescale_count
            if target < self.floor:
                raise ValueError('pending product exhausted native Boot input budget')
            self.normalized[x.id] = self.rescale(x, target, self.scales[target])
        return self.normalized[x.id]

    def safe(self, x):
        x = self.normalize(x)
        return self.boot(x, 'budget-exhausted') if x.level < self.floor + self.rescale_count else x

    def boot(self, x, label):
        x = self.normalize(x)
        # Close EVERY encoder block at its input level/scale, rather than
        # giving later blocks a larger hidden-state budget than the first.
        target = self.initial if fnmatch.fnmatchcase(label, '*.ffn_ln.output.*') else self.anchor
        key = (x.id, label if self.strategy == 'stage' else '', target)
        if key in self.refreshed:
            return self.refreshed[key]
        if x.level < self.floor:
            raise ValueError('bootstrap input below declared profile budget')
        # The historical scale30 profile needs a real 2^16 plaintext lift.
        # A scale~56 input already satisfies the normalization integer bound.
        if x.scale < 40:
            one = self.unity(x.level, 16.0)
            x = self.compute('mul_cp', [x, one], x.level, x.scale + one.scale)
        attrs = dict(target_level=self.profile['output_level'],
            target_scale_log2=self.profile['output_scale_log2'], target_components=2,
            operator_profile=self.profile['profile_id'], implementation='native')
        y = self.compute('boot', [x], self.profile['output_level'],
                         float(self.profile['output_scale_log2']), attrs=attrs)
        out = self.align(y, target)
        self.refreshed[key] = out
        self.boot_labels[label] += 1
        return out

    def lower_linear(self, n):
        operands = [self.safe(self.lowered[a.index]) for a in n.args]
        level = min(x.level for x in operands)
        operands = [self.align(x, level) for x in operands]
        label, output_tile = n.data
        matrix = self.graph.matrices[label]
        target = level - self.rescale_count
        call_id = len(self.prepared_calls)
        y = self.compute('ApiCall', operands, target, self.scales[target],
            attrs=dict(callee='bert.compact_linear', call_id=str(call_id)))
        call = dict(call_id=str(call_id), matrix_id=label, output_tile=output_tile,
            input_width=matrix['weight'].shape[1], output_width=matrix['weight'].shape[0],
            tile=matrix['tile'], baby_step=matrix['baby_step'], input_level=level,
            input_scale_log2=operands[0].scale, output_level=target,
            output_scale_log2=y.scale, rescale_count=self.rescale_count)
        self.prepared_calls.append(call)
        inputs = len(operands)
        terms = inputs * (2 * matrix['tile'] - 1)
        self.linear_stats['plaintext_products'] += terms
        self.linear_stats['rescale_operations'] += 1
        self.linear_stats['rescale_prime_drops'] += self.rescale_count
        return y

    def lower_qk(self, n):
        if self.rescale_count != 2:
            raise ValueError('prepared QK requires two-prime logical rescaling')
        operands = [self.safe(self.lowered[a.index]) for a in n.args]
        operands = [self.boot(x, 'qk-budget-frontier') if x.level < self.floor + 4 else x
                    for x in operands]
        level = min(x.level for x in operands)
        operands = [self.align(x, level) for x in operands]
        target = level - 4
        call_id = len(self.prepared_qk_calls)
        y = self.compute('ApiCall', operands, target, self.scales[target],
            attrs=dict(callee='bert.compact_qk', call_id=str(call_id)))
        spec = self.graph.qk_specs[n.data]
        self.prepared_qk_calls.append(dict(spec, call_id=str(call_id), id=n.data,
            input_level=level, input_scale_log2=operands[0].scale,
            intermediate_scale_log2=self.scales[level - 2], output_level=target,
            output_scale_log2=y.scale))
        tokens = spec['tokens']
        packed = spec.get('pack_inputs', False)
        products = tokens // 2 if packed else tokens
        self.qk_stats['cc_products'] += products
        self.qk_stats['relinearizations'] += products
        self.qk_stats['plaintext_products'] += tokens
        self.qk_stats['rescale2_operations'] += products + 1
        self.qk_stats['rescale_prime_drops'] += 2 * (products + 1)
        if packed:
            giants, pairs = tokens // spec['baby'], spec['baby'] // 2
            reductions = spec['head_dim'].bit_length() - 1
            adds = 2 + products * (reductions + 1) + giants * (pairs - 1) + giants - 1
            self.qk_stats['rotations'] += adds + giants + pairs - 2
            self.qk_stats['additions'] += adds
            self.qk_stats['hoisted_rotation_batches'] += 2
        return y

    def lower_av(self, n):
        scores, value = [self.safe(self.lowered[arg.index]) for arg in n.args]
        if self.rescale_count != 2 or (scores.level, value.level) != (10, 8) or self.floor > 4:
            raise ValueError('prepared AV backend requires physical scoreL10/valueL8 -> commonL6/outputL4')
        common, target = 6, 4
        call_id = len(self.prepared_av_calls)
        partial = n.op == 'compact_av_partial'
        y = self.compute('ApiCall', [scores, value], target, self.scales[target],
            attrs=dict(callee=AV_PARTIAL_CALLEE if partial else 'bert.compact_av', call_id=str(call_id)))
        spec = self.graph.av_specs[n.data]
        self.prepared_av_calls.append(dict(spec, call_id=str(call_id), id=n.data,
            score_input_level=scores.level, score_input_scale_log2=scores.scale,
            value_input_level=value.level, value_input_scale_log2=value.scale,
            common_level=common, common_scale_log2=self.scales[common],
            output_level=target, output_scale_log2=y.scale))
        tokens = spec['tokens']
        if partial:
            baby, dim = spec['baby'], spec['head_dim']
            giants, products = tokens // baby, tokens // 2
            self.av_stats.update(cc_products=products, relinearizations=products,
                plaintext_products=2 * tokens + dim - 1,
                rescale_operations=products + tokens + 1,
                rescale2_operations=products + 1, rescale4_operations=tokens,
                rescale_prime_drops=2 * (products + 1) + 4 * tokens,
                rotations=2 * tokens + baby + baby // 2 + 2 * giants - 4,
                additions=2 * tokens + dim - 1,
                hoisted_rotation_batches=int(baby > 1) + int(baby > 2) +
                    int(giants > 1) + giants * int(baby > 1),
                removed_score_alignment_products=tokens - 1)
        else:
            self.av_stats.update(cc_products=tokens, relinearizations=tokens,
                plaintext_products=3 * tokens - 1, rescale_operations=2 * tokens + 1,
                rescale2_operations=tokens + 1, rescale4_operations=tokens,
                rescale_prime_drops=6 * tokens + 2,
                removed_score_alignment_products=tokens - 1)
        return y

    def lower(self, outputs):
        outputs = list(outputs)
        validate_partial_av_consumers(self.graph, outputs)
        nodes = self.graph.reachable(outputs)
        references = Counter(arg.index for node in nodes for arg in node.args)
        for n in nodes:
            if n.op == 'plain':
                continue
            if n.op == 'input':
                host = self.value(self.initial, self.scales[self.initial], place=self.host)
                y = self.value(host.level, host.scale)
                self.external.append(host.id)
                self.transfer(host, y, self.initialization)
            elif n.op == 'compact_linear':
                y = self.lower_linear(n)
            elif n.op == 'compact_qk':
                y = self.lower_qk(n)
            elif n.op in ('compact_av', 'compact_av_partial'):
                y = self.lower_av(n)
            elif n.op == 'boot':
                x = self.lowered[n.args[0].index]
                keep = self.strategy == 'stage' or (self.strategy == 'selective' and
                    any(fnmatch.fnmatchcase(str(n.data), p) for p in self.SELECTIVE))
                y = self.boot(x, str(n.data)) if keep else x
            elif n.op in ('rotate', 'neg'):
                source = n.args[0]
                x = self.lowered[source.index]
                binary_mask = (source.op == 'mul' and source.args[1].op == 'plain' and
                               source.args[1].data in self.graph.binary_masks)
                shared = self.normalize_shared_rotations and references[source.index] > 1
                if n.op == 'rotate' and x.pending and (binary_mask or shared):
                    original = x
                    newly_normalized = x.id not in self.normalized
                    x = self.normalize(x)
                    if shared and not binary_mask and newly_normalized:
                        self.shared_rotation_frontiers.append(dict(graph_node=source.index,
                            source_level=original.level, normalized_level=x.level,
                            references=references[source.index]))
                y = self.compute('rotate' if n.op == 'rotate' else 'negate', [x], x.level, x.scale,
                    attrs=dict(steps=n.data) if n.op == 'rotate' else None, pending=x.pending)
            else:
                a, b = n.args
                if a.op == 'plain':
                    x = self.normalize(self.lowered[b.index])
                    p = self.plain(a.data, x.level, x.scale)
                    neg = self.compute('negate', [x], x.level, x.scale)
                    y = self.compute('add_cp', [neg, p], x.level, x.scale)
                elif b.op == 'plain':
                    x = self.safe(self.lowered[a.index]) if n.op == 'mul' else self.normalize(self.lowered[a.index])
                    p = self.plain(b.data, x.level, x.scale)
                    if n.op == 'mul':
                        y = self.compute('mul_cp', [x, p], x.level, 2 * x.scale, pending=True)
                        if not self.fuse:
                            y = self.normalize(y)
                    else:
                        y = self.compute(n.op + '_cp', [x, p], x.level, x.scale)
                else:
                    x, z = self.lowered[a.index], self.lowered[b.index]
                    raw_sum = (n.op in ('add', 'sub') and x.pending and z.pending and
                               x.level == z.level and abs(x.scale - z.scale) < 1e-8)
                    if not raw_sum:
                        x, z = ((self.safe(x), self.safe(z)) if n.op == 'mul'
                                else (self.normalize(x), self.normalize(z)))
                        level = min(x.level, z.level)
                        x, z = self.align(x, level), self.align(z, level)
                    if n.op == 'mul':
                        product = self.compute('mul_cc', [x, z], x.level, x.scale + z.scale, 3)
                        relin = self.compute('relinearize', [product], x.level, product.scale)
                        target = x.level - self.rescale_count
                        y = self.rescale(relin, target, self.scales[target])
                    else:
                        y = self.compute(n.op + '_cc', [x, z], x.level, x.scale, pending=raw_sum)
            self.lowered[n.index] = y
        return [self.normalize(self.lowered[x.index]) for x in outputs]

    def export(self, outputs, directory, resident=False):
        outputs = list(outputs)
        validate_partial_av_consumers(self.graph, outputs)
        stats = super().export(outputs, directory, resident=resident)
        directory = Path(directory)
        # Prepared calls are explicitly versioned and never disguised as
        # ordinary scalar arithmetic by either the plan or this report.
        import json
        path = directory / 'model.runtime-plan.json'
        plan = json.loads(path.read_text())
        plan['format_version'] = 7
        if not self.encoded:
            # A prepared-only Linear has no standard Encode payloads.
            # The verifier requires bundle presence iff a bundle Encode exists.
            plan.pop('plaintext_bundle', None)
        path.write_bytes(json_bytes(plan))
        try:
            from .layer_observability import collect_layer_boundaries
        except ImportError:
            from layer_observability import collect_layer_boundaries
        boundaries = collect_layer_boundaries(self.graph, self, plan, outputs)
        (directory / 'layer-boundaries.json').write_bytes(json_bytes(boundaries))
        stats['layer_boundary_count'] = len(boundaries['layers'])
        matrices = []
        offset = 0
        rotations = set()
        with (directory / 'linear-weights.bin').open('wb') as stream:
            for label, matrix in getattr(self.graph, 'matrices', {}).items():
                raw_weight = matrix['weight'].tobytes()
                raw_bias = matrix['bias'].tobytes()
                stream.write(raw_weight)
                stream.write(raw_bias)
                matrices.append(dict(matrix_id=label, input_width=matrix['weight'].shape[1],
                    output_width=matrix['weight'].shape[0], weight_offset=offset,
                    bias_offset=offset + len(raw_weight), tile=matrix['tile'],
                    baby_step=matrix['baby_step']))
                offset += len(raw_weight) + len(raw_bias)
                baby = min(matrix['baby_step'], 2 * matrix['tile'] - 1)
                rotations.update(range(1, baby))
                rotations.update(g * baby for g in range((1 - matrix['tile']) // baby,
                    (matrix['tile'] - 1) // baby + 1) if g)
        manifest = dict(format_version=1, dtype='<f4', weight_file='linear-weights.bin',
            weight_sha256=digest((directory / 'linear-weights.bin').read_bytes()),
            matrices=matrices, calls=self.prepared_calls, rotation_steps=sorted(rotations))
        (directory / 'compact-linear.json').write_bytes(json_bytes(manifest))
        if self.prepared_qk_calls:
            steps = set()
            for call in self.prepared_qk_calls:
                baby, tokens, tile = call['baby'], call['tokens'], call['tile']
                if call.get('pack_inputs', False):
                    steps.update((-call['head_dim'], call['head_dim'] - 1,
                                  tile - call['head_dim']))
                for b in range(1, baby):
                    steps.update((tile * b, -b))
                for giant in range(1, tokens // baby):
                    steps.update((-tile * baby * giant, (tile * baby - baby) * giant))
                shift = 1
                while shift < call['head_dim']:
                    steps.add(shift)
                    shift *= 2
            slots = self.graph.slots
            steps = {((step % slots) - slots if step % slots > slots // 2 else step % slots)
                     for step in steps if step % slots}
            # v2 is a capability gate: legacy v1 readers must reject a request
            # they would otherwise silently execute with the baseline schedule.
            qk_version = 2 if any(call.get('pack_inputs', False) or call.get('pair_products', False)
                                  for call in self.prepared_qk_calls) else 1
            (directory / 'prepared-qk.json').write_bytes(json_bytes(dict(format_version=qk_version,
                calls=self.prepared_qk_calls, rotation_steps=sorted(steps))))
            stats['prepared_qk_calls'] = len(self.prepared_qk_calls)
            stats['prepared_qk_arithmetic'] = dict(self.qk_stats)
            if any(call.get('pack_inputs', False) for call in self.prepared_qk_calls):
                stats['qk_pack_inputs'] = dict(enabled=True, schedule='packed_input2_pre_cc_gap',
                    arithmetic_scope='static planned internals, not observed GPU evaluator calls',
                    ideal_message_zero_gaps_proven=True,
                    logical_key_offsets_per_pair=self.prepared_qk_calls[0]['tokens'],
                    note='same full QK polynomial on proven ideal zero gaps; CKKS gap noise is not bitwise zero and noise ordering can differ')
        if self.prepared_av_calls:
            steps = set()
            for call in self.prepared_av_calls:
                baby, tokens, tile = call['baby'], call['tokens'], call['tile']
                if call.get('algorithm') == AV_PARTIAL_ALGORITHM:
                    steps.update((-call['head_dim'], call['head_dim'], tile - call['head_dim']))
                giants = tokens // baby
                steps.update(tile * b for b in range(1, baby))
                for giant in range(1, giants):
                    steps.update((tile * baby * giant, (baby - tile * baby) * giant))
                steps.update(range(1, baby))
                steps.update(b - tokens for b in range(0 if giants > 1 else 1, baby))
            slots = self.graph.slots
            steps = {((step % slots) - slots if step % slots > slots // 2 else step % slots)
                     for step in steps if step % slots}
            av_version = 2 if any(call.get('algorithm') == AV_PARTIAL_ALGORITHM
                                  for call in self.prepared_av_calls) else 1
            (directory / 'prepared-av.json').write_bytes(json_bytes(dict(format_version=av_version,
                calls=self.prepared_av_calls, rotation_steps=sorted(steps))))
            stats['prepared_av_calls'] = len(self.prepared_av_calls)
            stats['prepared_av_arithmetic'] = dict(self.av_stats)
        stats.update(physical_rescale_count=self.rescale_count,
            working_scale_log2=self.scales[0],
            final_levels=[self.values[int(i)]['level'] for i in plan['final_outputs']],
            encoder_boundary_level=(self.initial if all(self.values[int(i)]['level'] == self.initial
                                    for i in plan['final_outputs']) else None),
            prepared_linear_calls=len(self.prepared_calls),
            prepared_linear_arithmetic=dict(self.linear_stats), dense_weight_bytes=offset,
            boot_input_scale_lift_plaintext_log2=16 if self.scales[0] < 40 else 0,
            prepared_backend='bench/dsl/bert/compact_native; true GPU tiled dense BSGS')
        if self.normalize_shared_rotations:
            stats['shared_rotation_rescale'] = dict(enabled=True,
                frontiers=self.shared_rotation_frontiers,
                note='real rescale before reused rotations; CKKS rounding/noise ordering can differ')
        (directory / 'schedule.json').write_bytes(json_bytes(stats))
        return stats
