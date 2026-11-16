"""Small expression frontend and explicit, prime-aware CKKS RuntimePlan lowering.

This deliberately bypasses automatic scale/Boot placement, not CKKS execution.
All refreshes in emitted plans are native GPU bootstraps. No secret key refresh.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import fnmatch
import json
import math
from pathlib import Path
import numpy as np


def digest(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()


@dataclass(eq=False)
class Expr:
    backend: 'Graph'
    index: int
    op: str
    args: tuple = ()
    data: object = None

    def __add__(self, other): return self.backend.binary('add', self, other)
    def __radd__(self, other): return self.backend.binary('add', other, self)
    def __sub__(self, other): return self.backend.binary('sub', self, other)
    def __rsub__(self, other): return self.backend.binary('sub', other, self)
    def __mul__(self, other): return self.backend.binary('mul', self, other)
    def __rmul__(self, other): return self.backend.binary('mul', other, self)
    def __neg__(self): return self.backend.node('neg', (self,))
    def rotate(self, step):
        step = int(step) % self.backend.slots
        return self if not step else self.backend.node('rotate', (self,), step)


class Graph:
    def __init__(self, slots):
        self.slots = slots
        self.nodes = []
        self.constants = {}
        self.binary_masks = set()
        self.cache = {}
        self.inputs = []

    def node(self, op, args=(), data=None):
        key = (op, tuple(a.index for a in args), data)
        if op not in ('input', 'boot') and key in self.cache:
            return self.cache[key]
        result = Expr(self, len(self.nodes), op, args, data)
        self.nodes.append(result)
        if op not in ('input', 'boot'): self.cache[key] = result
        return result

    def input(self, name):
        result = self.node('input', data=name)
        self.inputs.append(result)
        return result

    def plain(self, values):
        a = np.asarray(values, dtype='<f8').reshape(-1)
        if not 0 < len(a) <= self.slots or not np.isfinite(a).all():
            raise ValueError('plaintext must contain finite real slots')
        # Short payloads mean a zero tail, never implicit repetition.
        nz = np.flatnonzero(a)
        a = a[:int(nz[-1]) + 1] if len(nz) else a[:1]
        content = digest(a.tobytes())
        if content not in self.constants:
            self.constants[content] = a.copy()
            if np.all((a == 0) | (a == 1)): self.binary_masks.add(content)
        return self.node('plain', data=content)

    def binary(self, op, lhs, rhs):
        if not isinstance(lhs, Expr): lhs = self.plain(np.full(self.slots, lhs))
        if not isinstance(rhs, Expr): rhs = self.plain(np.full(self.slots, rhs))
        if lhs.backend is not self or rhs.backend is not self:
            raise ValueError('expressions belong to different graphs')
        if lhs.op == rhs.op == 'plain':
            a = np.zeros(self.slots); b = np.zeros(self.slots)
            av = self.constants[lhs.data]; bv = self.constants[rhs.data]
            a[:len(av)] = av; b[:len(bv)] = bv
            return self.plain({'add': np.add, 'sub': np.subtract, 'mul': np.multiply}[op](a, b))
        if lhs.op == 'plain' and op in ('add', 'mul'): lhs, rhs = rhs, lhs
        # Fuse adjacent binary masks. Do NOT fold a mask into just the zero
        # baby-rotation's dense diagonal: that unbalances BSGS term depths and
        # prevents sharing their final rescale.
        if (op == 'mul' and rhs.op == 'plain' and lhs.op == 'mul' and
                lhs.args[1].op == 'plain' and rhs.data in self.binary_masks and
                lhs.args[1].data in self.binary_masks):
            combined = self.binary('mul', lhs.args[1], rhs)
            return self.binary('mul', lhs.args[0], combined)
        return self.node(op, (lhs, rhs))

    def bootstrap(self, expr, label='stage'):
        return self.node('boot', (expr,), label)

    def reachable(self, outputs):
        todo = list(outputs); seen = set()
        while todo:
            node = todo.pop()
            if node.index in seen: continue
            seen.add(node.index); todo.extend(node.args)
        return [n for n in self.nodes if n.index in seen]

    def evaluate(self, outputs, inputs):
        """Slot arithmetic oracle; does not simulate CKKS noise or bootstrap error."""
        vals = {}
        for n in self.reachable(outputs):
            args = [vals[a.index] for a in n.args]
            if n.op == 'input': v = np.asarray(inputs[n.data])
            elif n.op == 'plain':
                v = np.zeros(self.slots); a = self.constants[n.data]; v[:len(a)] = a
            elif n.op == 'boot': v = args[0]
            elif n.op == 'rotate': v = np.roll(args[0], -n.data)
            elif n.op == 'neg': v = -args[0]
            else: v = {'add': np.add, 'sub': np.subtract, 'mul': np.multiply}[n.op](*args)
            vals[n.index] = v
        return [vals[o.index] for o in outputs]


def coefficient_moduli(degree, bits):
    """Match CoeffModulus::Create's descending search and reverse consumption."""
    def prime(n):
        if n < 2: return False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if n % p == 0: return n == p
        d = n - 1; s = 0
        while d % 2 == 0: d //= 2; s += 1
        for a in (2, 3, 5, 7, 11):  # deterministic for these <=32-bit moduli
            x = pow(a, d, n)
            if x in (1, n - 1): continue
            for _ in range(s - 1):
                x = x * x % n
                if x == n - 1: break
            else: return False
        return True
    table = {}
    for b, count in Counter(bits).items():
        q = ((1 << b) - 1) // (2 * degree) * (2 * degree) + 1
        found = []
        while len(found) < count and q > (1 << (b - 1)):
            if prime(q): found.append(q)
            q -= 2 * degree
        if len(found) != count: raise ValueError('not enough NTT primes')
        table[b] = found
    return [table[b].pop() for b in bits]


@dataclass(frozen=True)
class Physical:
    id: str
    level: int
    scale: float
    components: int = 2
    pending: bool = False


class ManualLowering:
    """Canonical scales avoid the exponential drift of rounded log2 scales.

    S[l-1] = 2*S[l] - log2(q[l]); compute S backwards from the bottom.
    Shorter residual branches are matched by multiply-one plus exact rescale,
    not by changing ciphertext metadata. Linear product sums are accumulated
    before one rescale. Boot output is matched using the same arithmetic.
    """
    def __init__(self, graph, context, profile, strategy='budget', fuse=True, initial_level=None):
        if strategy not in ('budget', 'stage', 'selective'): raise ValueError(strategy)
        self.graph = graph; self.context = context; self.profile = profile
        self.strategy = strategy; self.fuse = fuse
        qbits = context['q_moduli_bits']; pbits = context['p_moduli_bits']
        primes = coefficient_moduli(context['poly_degree'], qbits + pbits)
        self.q = primes[:len(qbits)]
        self.anchor = profile['output_level'] - 1
        self.initial = self.anchor if initial_level is None else int(initial_level)
        if not 0 <= self.initial < len(self.q): raise ValueError('initial level outside Q chain')
        self.floor = max(2, profile.get('input_level_min', 2))
        if self.anchor <= self.floor + 2: raise ValueError('insufficient native Boot output budget')
        self.scales = [float(context['default_scale_log2'])]
        for q in self.q[1:]: self.scales.append((self.scales[-1] + math.log2(q)) / 2)
        self.values = []; self.initialization = []; self.execution = []
        self.lowered = {}; self.normalized = {}; self.refreshed = {}; self.aligned = {}
        self.encoded = {}; self.external = []; self.transfer_id = 0
        self.device = dict(kind='device', rank=0, index=0)
        self.host = dict(kind='host', rank=0)
        self.boot_labels = Counter()

    def value(self, level, scale, components=2, place=None, pending=False):
        ident = str(len(self.values))
        self.values.append(dict(id=ident, kind='plaintext' if components == 1 else 'ciphertext',
            place=self.device if place is None else place, context='bert-research', level=level,
            scale_log2=scale, ntt=True, components=components))
        return Physical(ident, level, scale, components, pending)

    def compute(self, op, inputs, level, scale, components=2, attrs=None, pending=False):
        out = self.value(level, scale, components, pending=pending)
        row = dict(kind='compute', op=op, inputs=[x.id for x in inputs], output=out.id, place=self.device)
        if attrs is not None: row['attrs'] = attrs
        self.execution.append(row)
        return out

    def transfer(self, src, dst, phase):
        phase.append(dict(kind='transfer', transfer_id=str(self.transfer_id), hint='point_to_point',
            inputs=[src.id], outputs=[dst.id], sources=[self.values[int(src.id)]['place']],
            destinations=[self.values[int(dst.id)]['place']],
            output_kinds=['plaintext' if dst.components == 1 else 'ciphertext']))
        self.transfer_id += 1

    def plain(self, content, level, scale):
        key = (content, level, round(scale, 12))
        if key in self.encoded: return self.encoded[key]
        host = self.value(level, scale, 1, self.host)
        dev = self.value(level, scale, 1)
        self.execution.append(dict(kind='encode', output=host.id, payload=dict(kind='bundle', content=content)))
        self.transfer(host, dev, self.execution)
        self.encoded[key] = dev
        return dev

    def unity(self, level, scale):
        content = self.graph.plain(np.ones(self.graph.slots)).data
        return self.plain(content, level, scale)

    def rescale(self, x, target, scale):
        return self.compute('rescale', [x], target, scale,
            attrs=dict(target_level=target, target_scale_log2=scale))

    def normalize(self, x):
        if not x.pending: return x
        if x.id not in self.normalized:
            self.normalized[x.id] = self.rescale(x, x.level - 1, self.scales[x.level - 1])
        return self.normalized[x.id]

    def boot(self, x, label):
        x = self.normalize(x)
        key = (x.id, label if self.strategy == 'stage' else '')
        if key in self.refreshed: return self.refreshed[key]
        if x.level < self.profile.get('input_level_min', 2):
            raise ValueError(f'bootstrap input too low: {x}')
        # Native ModRaise normalization multiplies by q0*q_next/scale.
        # At canonical ~30-bit scale this exceeds its exact 2^53 integer
        # bound. A REAL multiply by encoded 1 at scale 2^16 raises the input
        # scale to ~46 bits (without changing its decoded value or its level).
        # No metadata-only scale rewrite is used.
        if x.scale < 40:
            one = self.unity(x.level, 16.0)
            x = self.compute('mul_cp', [x, one], x.level, x.scale+one.scale)
        attrs = dict(target_level=self.profile['output_level'],
            target_scale_log2=self.profile['output_scale_log2'], target_components=2,
            operator_profile=self.profile['profile_id'], implementation='native')
        y = self.compute('boot', [x], self.profile['output_level'],
                         float(self.profile['output_scale_log2']), attrs=attrs)
        out = self.align(y, self.anchor)
        self.refreshed[key] = out; self.boot_labels[label] += 1
        return out

    def safe(self, x):
        x = self.normalize(x)
        return self.boot(x, 'budget-exhausted') if x.level <= self.floor else x

    def align(self, x, target):
        x = self.normalize(x)
        if x.level == target: return x
        if x.level < target: raise ValueError('cannot modraise without a bootstrap')
        key = (x.id, target)
        if key not in self.aligned:
            scale = self.scales[target]
            drop = sum(math.log2(q) for q in self.q[target + 1:x.level + 1])
            p = self.unity(x.level, scale + drop - x.scale)
            product = self.compute('mul_cp', [x, p], x.level, x.scale + p.scale)
            self.aligned[key] = self.rescale(product, target, scale)
        return self.aligned[key]

    def lower(self, outputs):
        for n in self.graph.reachable(outputs):
            if n.op == 'plain': continue
            if n.op == 'input':
                h = self.value(self.initial, self.scales[self.initial], place=self.host)
                d = self.value(h.level, h.scale)
                self.external.append(h.id); self.transfer(h, d, self.initialization)
                y = d
            elif n.op == 'boot':
                x = self.lowered[n.args[0].index]
                # SIMD cuts protect shared branch frontiers, not each of the
                # 128 exp branches. Keep only cuts that bridge long intervals.
                selective = ('*.attention.denominator.*', '*.attention.weighted_value.*',
                    '*.attention_residual.*', '*.attention_ln.output.*', '*.gelu.*',
                    '*.ffn_ln.variance.*', '*.ffn_ln.output.*')
                keep = self.strategy == 'stage' or (self.strategy == 'selective' and
                    any(fnmatch.fnmatchcase(str(n.data), pattern) for pattern in selective))
                y = self.boot(x, str(n.data)) if keep else x
            elif n.op in ('rotate', 'neg'):
                source = n.args[0]
                x = self.lowered[source.index]
                if (n.op == 'rotate' and x.pending and source.op == 'mul' and
                        source.args[1].op == 'plain' and source.args[1].data in self.graph.binary_masks):
                    # Rescale a shared masked source ONCE before its baby
                    # rotations, rather than once per rotated descendant.
                    x = self.normalize(x)
                y = self.compute('rotate' if n.op == 'rotate' else 'negate', [x], x.level, x.scale,
                    attrs=dict(steps=n.data) if n.op == 'rotate' else None, pending=x.pending)
            else:
                a, b = n.args
                if a.op == 'plain':
                    # p - x: preserve the real expression, do not silently commute subtraction.
                    x = self.normalize(self.lowered[b.index])
                    p = self.plain(a.data, x.level, x.scale)
                    neg = self.compute('negate', [x], x.level, x.scale)
                    y = self.compute('add_cp', [neg, p], x.level, x.scale)
                elif b.op == 'plain':
                    x = self.safe(self.lowered[a.index]) if n.op == 'mul' else self.normalize(self.lowered[a.index])
                    p = self.plain(b.data, x.level, x.scale)
                    if n.op == 'mul':
                        y = self.compute('mul_cp', [x, p], x.level, 2*x.scale, pending=True)
                        if not self.fuse: y = self.normalize(y)
                    else: y = self.compute(n.op + '_cp', [x, p], x.level, x.scale)
                else:
                    x, z = self.lowered[a.index], self.lowered[b.index]
                    raw_sum = (n.op in ('add', 'sub') and x.pending and z.pending and
                               x.level == z.level and abs(x.scale-z.scale) < 1e-8)
                    if not raw_sum:
                        x, z = (self.safe(x), self.safe(z)) if n.op == 'mul' else (self.normalize(x), self.normalize(z))
                        level = min(x.level, z.level)
                        x, z = self.align(x, level), self.align(z, level)
                    if n.op == 'mul':
                        product = self.compute('mul_cc', [x, z], x.level, x.scale+z.scale, 3)
                        relin = self.compute('relinearize', [product], x.level, product.scale)
                        y = self.rescale(relin, x.level-1, self.scales[x.level-1])
                    else:
                        y = self.compute(n.op+'_cc', [x, z], x.level, x.scale, pending=raw_sum)
            self.lowered[n.index] = y
        return [self.normalize(self.lowered[x.index]) for x in outputs]

    def export(self, outputs, directory, resident=False):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
        results = self.lower(outputs)
        spec = make_spec(self.context, self.profile, self.q)
        spec_bytes = json_bytes(spec); (directory/'operator-spec.json').write_bytes(spec_bytes)
        bundle = directory/'bundle'; bundle.mkdir()
        contents = {key[0] for key in self.encoded}
        blobs = []; offset = 0
        with (bundle/'data.bin').open('wb') as f:
            for content in sorted(contents):
                raw = self.graph.constants[content].tobytes(); f.write(raw)
                blobs.append(dict(content=content, byte_length=len(raw), offset=offset)); offset += len(raw)
        manifest = dict(bundle_format_version=2, bundle_id='bert-plaintexts', version=1,
                        blobs=blobs, pack_byte_length=offset)
        raw = json_bytes(manifest); (bundle/'manifest.json').write_bytes(raw)
        if resident:
            setup = []; online = []
            for row in self.execution:
                if row['kind'] == 'encode' or (row['kind'] == 'transfer' and row['output_kinds'] == ['plaintext']):
                    setup.append(row)
                else: online.append(row)
            self.initialization.extend(setup); self.execution = online
        final = [x.id for x in results]
        # Last-use release; persistent plaintexts survive setup but not the iteration.
        all_rows = self.initialization + self.execution
        last = {}
        for i, row in enumerate(all_rows):
            for ident in row.get('inputs', []): last[ident] = i
        release = {}
        for ident, i in last.items():
            if ident not in final and ident not in self.external:
                release.setdefault(i, []).append(dict(kind='release', value=ident))
        init = []; execution = []
        for i, row in enumerate(all_rows):
            dest = init if i < len(self.initialization) else execution
            dest.append(row); dest.extend(release.get(i, []))
        for i, row in enumerate(init+execution): row['ordinal'] = i
        plan = dict(format_version=5, plan_id='1', target=dict(target_id='poseidon-ckks-gpu',
            world_size=1, device_counts=[1], capability_version=1,
            operator_spec=dict(id=spec['spec_id'], version=1, source_sha256=digest(spec_bytes))),
            plaintext_bundle=dict(id='bert-plaintexts', version=1, manifest_sha256=digest(raw)),
            values=self.values, external_inputs=self.external, initialization=init,
            execution=execution, finalization=[], final_outputs=final)
        (directory/'model.runtime-plan.json').write_bytes(json_bytes(plan))
        counter = Counter(row.get('op', row['kind']) for row in all_rows)
        encoded_bytes = sum((int(k[1])+1)*self.context['poly_degree']*4 for k in self.encoded)
        memory = memory_sweep(plan, self.context['poly_degree'])
        stats = dict(strategy=self.strategy, fused_linear_rescales=self.fuse,
            resident_plaintexts=resident, instruction_counts=dict(counter),
            boot_labels=dict(self.boot_labels), plaintext_raw_bytes=offset,
            estimated_encoded_plaintext_bytes=encoded_bytes, graph_nodes=len(self.graph.nodes),
            values=len(self.values), input_count=len(self.external), output_count=len(final),
            memory_lower_bounds=memory,
            canonical_scales=self.scales, initial_level=self.initial, boot_normalized_level=self.anchor,
            boot_input_scale_lift_plaintext_log2=16,
            initial_scale_log2=self.scales[self.initial], actual_q_moduli=self.q,
            lowering='explicit Python CKKS RuntimePlan emitter; automatic compiler planner bypassed',
            security='none; research-only; accuracy not validated')
        (directory/'schedule.json').write_bytes(json_bytes(stats))
        return stats


def memory_sweep(plan, degree):
    """Payload-only liveness; excludes keys, native Boot and evaluator scratch."""
    descriptions = {v['id']: v for v in plan['values']}
    live = set()
    current = Counter(); peaks = Counter()
    def allocate(ident):
        if ident in live: raise ValueError('value defined twice: '+ident)
        live.add(ident)
        v = descriptions[ident]
        key = v['place']['kind'] + '_' + v['kind']
        current[key] += v['components'] * (v['level']+1) * degree * 4
        for name,value in current.items(): peaks[name] = max(peaks[name], value)
        peaks['device_total'] = max(peaks['device_total'], current['device_plaintext']+current['device_ciphertext'])
    for ident in plan['external_inputs']: allocate(ident)
    for row in plan['initialization']+plan['execution']+plan['finalization']:
        if row['kind'] == 'release':
            ident=row['value']
            if ident not in live: raise ValueError('release of a non-live value: '+ident)
            live.remove(ident); v=descriptions[ident]
            current[v['place']['kind']+'_'+v['kind']] -= v['components']*(v['level']+1)*degree*4
            continue
        for ident in row.get('inputs', []):
            if ident not in live: raise ValueError('use after release: '+ident)
        for ident in row.get('outputs', [row.get('output')]):
            if ident is not None: allocate(ident)
    for ident in plan['final_outputs']:
        if ident not in live: raise ValueError('final output not live: '+ident)
    return dict(peak_payload_bytes=dict(peaks),
        excludes='evaluation keys, native Boot constants/workspace, allocator reserve, internal evaluator temporaries')


def make_spec(context, profile, primes):
    count = len(primes)
    costs = dict(add_cc=10, add_cp=10, sub_cc=10, sub_cp=10, mul_cc=50,
                 mul_cp=20, negate=5, rotate=100, rescale=20, mod_switch=10, relinearize=70)
    operators = {name: dict(supported=True, latency_us_by_level=[cost]*count, noise_by_level=None)
                 for name, cost in costs.items()}
    operators['rescale']['max_levels_per_op'] = count
    operators['boot'] = dict(supported=True)
    boot = dict(profile, implementation='native', input_components=2, output_components=2,
                latency_us_by_input_level=[300000]*count, noise_by_input_level=None,
                host_requirements=dict(needs_secret_key=False, needs_host_compute=False))
    return dict(spec_format_version=2, spec_id='bert-a800-research-v1', version=1,
        status='validated', target_id='poseidon-ckks-gpu', rescale_mode='native', noise_unit=None,
        provenance=dict(kind='research-manual-schedule', repository='poseidon', revision='uncommitted',
                        path='bench/bert_dsl/backend.py', source_sha256=digest(Path(__file__).read_bytes())),
        context=dict(context_id='bert-research', poly_degree=context['poly_degree'],
            rns_moduli_log2=context['q_moduli_bits'], rns_moduli=primes,
            max_modulus_log2=max(context['q_moduli_bits']), default_scale_log2=context['default_scale_log2']),
        levels=dict(lower_bound=0, upper_bound=count-1), operators=operators, boot_profiles=[boot])
