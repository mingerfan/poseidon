"""Generic staged mixed packed / replicated-cell trusted rule baseline.

Canonical C-order chunks use the existing bound public Linear implementation.
Views only rearrange immutable logical references. Cross-slot arithmetic lazily
extracts individual replicated cells; cells remain cells until a Linear or the
physical output boundary needs canonical chunks. No Provider answer is rewritten.
No depth, allocation, CKKS accuracy or backend admission is claimed here.
"""
import copy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import numpy as np
from benchmark_graph import require, validate
from unified_graph_contract import validate_request, validate_candidate
from unified_graph_lowering import Scalar, power_polynomial
import unified_capacity_constants as capacity_constants
import unified_capacity_lowering as capacity_lowering
from unified_capacity_lowering import Chunks, PublicVector, vector_arithmetic, public_vector
from unified_capacity_layout import ABIS
from benchmarks.tools.staged_replicated_scalar_rule_v1 import replicated_scalar_lowering as scalar_rule

VERSION = "poseidon-mixed-packed-scalar-rule-v1"
FORMAT = "poseidon-mixed-packed-scalar-rule-record-v1"
SOURCE_KIND = scalar_rule.SOURCE_KIND
canonical = scalar_rule.canonical
digest = scalar_rule.digest
exact = scalar_rule.exact


class MixedEmitter(scalar_rule.ReplicatedEmitter):
    """Literal authority is the existing public-origin rule, reconstructed lazily."""
    def __init__(self, request, checked):
        super().__init__(request, {})
        self.checked = checked
        self.authority_loaded = False
        self.capacity_inline_rotations = True
        self.capacity_helpers = {}
        self.emitted_statement_bytes = 0

    def emit(self, text, zero=False):
        result = super().emit(text, zero)
        self.emitted_statement_bytes += len(self.lines[-1].encode())+1
        require(self.emitted_statement_bytes <= 65536, "Mixed emitted statement source budget")
        return result

    def literal(self, value):
        key = exact(float(value))
        if key not in self.scalar_names and not self.authority_loaded:
            self.allowed = scalar_rule.literal_sources(self.request["model"], self.checked)
            self.authority_loaded = True
        return super().literal(value)


@dataclass(frozen=True)
class PackedStorage:
    key: int
    parts: tuple
    elements: int


@dataclass(frozen=True)
class BoundCell:
    storage: PackedStorage
    index: int


class TensorValue:
    """A logical ndarray of immutable BoundCell or replicated Scalar references."""
    def __init__(self, entries):
        self.entries = np.asarray(entries, dtype=object)
        require(all(isinstance(v, (BoundCell, Scalar)) for v in self.entries.flat),
                "Mixed value must retain cipher logical references")
        self.entries.setflags(write=False)
        self.shape = tuple(self.entries.shape)

    def canonical_storage(self):
        flat = self.entries.reshape(-1)
        if not flat.size or not isinstance(flat[0], BoundCell):
            return None
        storage = flat[0].storage
        if storage.elements != flat.size:
            return None
        if not all(isinstance(ref, BoundCell) and ref.storage is storage and ref.index == i
                   for i, ref in enumerate(flat)):
            return None
        return storage


class Materializer:
    """Emitter-local immutable index maps, exact cell cache and pack boundaries."""
    def __init__(self, emitter):
        self.em = emitter
        self.storages = []
        self.cells = {}
        self.pack_cache = {}
        self.materializations = []
        self.packs = []
        p = emitter.period
        require(type(p) is int and p > 0 and p & (p-1) == 0, "Mixed power-of-two period")
        require(emitter.request["public_constants"].get("mask0") == [1.]+[0.]*(p-1),
                "Mixed bound mask0 must be full-period one-hot")

    def packed_value(self, parts, shape, *, producer):
        size = math.prod(shape)
        require(size > 0 and len(parts) == math.ceil(size/self.em.period), "Mixed canonical chunk count")
        require(all(isinstance(part, Scalar) and part.e is self.em for part in parts), "Mixed foreign emitter")
        storage = PackedStorage(len(self.storages), tuple(parts), size)
        self.storages.append(dict(storage=storage.key, producer=copy.deepcopy(producer), shape=list(shape),
                                 elements=size, parts=[part.name for part in parts]))
        refs = np.empty(size, dtype=object)
        for index in range(size):
            refs[index] = BoundCell(storage, index)
        return TensorValue(refs.reshape(shape))

    def cell(self, ref):
        if isinstance(ref, Scalar):
            require(ref.e is self.em, "Mixed foreign replicated cell")
            return ref
        require(isinstance(ref, BoundCell) and type(ref.index) is int and
                0 <= ref.index < ref.storage.elements, "Mixed invalid logical cell")
        chunk, slot = divmod(ref.index, self.em.period)
        original = ref.storage.parts[chunk]
        key = (original.name, slot, self.em.period)
        if key in self.cells:
            return self.cells[key]
        mask = "mask"+str(slot)
        if mask in self.em.request["public_constants"]:
            require(self.em.request["public_constants"][mask] ==
                    [float(i == slot) for i in range(self.em.period)], "Mixed bound cell mask is not one-hot")
            masked = self.em.zero if original.zero else self.em.emit(original.name+" * "+mask)
            route = "bound_slot_mask"
        else:
            masked = self.em.zero if original.zero else self.em.emit(self.em.rotate(original, slot).name+" * mask0")
            mask, route = "mask0", "bound_mask0_rotated_source"
        value = masked
        if not value.zero:
            for bit in range(self.em.period.bit_length()-1):
                value = value+self.em.rotate(value, 1 << bit)
        self.cells[key] = value
        self.materializations.append(dict(storage=ref.storage.key, logical_index=ref.index, chunk=chunk,
            slot=slot, source=original.name, mask=mask, route=route, replicated_expression=value.name))
        return value

    def array(self, value):
        if not isinstance(value, TensorValue):
            return value
        result = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            result[index] = self.cell(value.entries[index])
        return result

    def chunks(self, value, *, boundary):
        require(isinstance(value, TensorValue), "Mixed packing requires cipher value")
        storage = value.canonical_storage()
        if storage is not None:
            return Chunks(self.em, storage.parts, value.shape)
        array = self.array(value).reshape(-1)
        key = tuple((item.name, item.zero) for item in array)
        if key not in self.pack_cache:
            parts = tuple(self.em.pack(array[i:i+self.em.period]) for i in range(0, array.size, self.em.period))
            self.pack_cache[key] = parts
            self.packs.append(dict(boundary=copy.deepcopy(boundary), shape=list(value.shape),
                elements=array.size, cell_expressions=[v.name for v in array], parts=[v.name for v in parts],
                packing="canonical_C_order_prefixes"))
        return Chunks(self.em, self.pack_cache[key], value.shape)

    def index_map(self, value):
        if not isinstance(value, TensorValue):
            return None
        result = []
        for ref in value.entries.flat:
            if isinstance(ref, BoundCell):
                result.append(["packed", ref.storage.key, ref.index])
            else:
                result.append(["replicated_cell", ref.name, ref.zero])
        return dict(shape=list(value.shape), binding=digest(result))


def view(op, xs, attrs):
    """Shape/index transformations emit no cipher arithmetic."""
    entries = [x.entries if isinstance(x, TensorValue) else x for x in xs]
    x = entries[0]
    if op == "reshape": result = x.reshape(attrs["shape"])
    elif op == "flatten": result = x.reshape(-1)
    elif op == "transpose": result = np.swapaxes(x, attrs["dim0"], attrs["dim1"])
    elif op == "permute": result = x.transpose(attrs["dims"])
    elif op == "rotate": result = np.roll(x.reshape(-1), -attrs["step"]).reshape(x.shape)
    elif op == "concat": result = np.concatenate(entries, axis=attrs["axis"])
    elif op == "stack": result = np.stack(entries, axis=attrs["axis"])
    elif op == "slice":
        index = [slice(None)]*x.ndim
        index[attrs["axis"]] = slice(attrs["start"], attrs["stop"], attrs["step"])
        result = x[tuple(index)]
    elif op == "split": result = np.split(x, np.cumsum(attrs["sections"])[:-1], axis=attrs["axis"])
    else: raise ValueError("Unsupported mixed view "+op)
    is_cipher = any(isinstance(value, TensorValue) for value in xs)
    if is_cipher:
        return [TensorValue(item) for item in result] if op == "split" else TensorValue(result)
    return result


def packed_binary(op, xs, outshape, node, packed, materializer):
    """Only identical canonical slot coordinates or registered public vectors."""
    em = materializer.em
    require(any(isinstance(value, TensorValue) for value in xs),
            "Mixed arithmetic requires cipher operand; all-public nodes unsupported")
    public = []
    operands = []
    count = math.ceil(math.prod(outshape)/em.period)
    for number, value in enumerate(xs):
        if isinstance(value, TensorValue):
            storage = value.canonical_storage()
            if storage is None or value.shape != tuple(outshape):
                return None
            operands.append(storage.parts)
            public.append(False)
        else:
            try: expanded = np.broadcast_to(value, outshape)
            except ValueError: return None
            if not expanded.size:
                return None
            if np.all(expanded == expanded.reshape(-1)[0]):
                operands.append((float(expanded.reshape(-1)[0]),)*count)
            else:
                # Authority is the original node/operand/chunk binding, never a
                # numeric lookup in opaque pv vectors.
                operands.append(tuple(public_vector(em, packed, node["id"], "elementwise", operand=number, chunk=j)
                                      for j in range(count)))
            public.append(True)
    if all(public):
        return None
    parts = []
    for a, b in zip(*operands):
        if isinstance(a, (Scalar, PublicVector)) and isinstance(b, (Scalar, PublicVector)):
            result = vector_arithmetic(em, op, a, b)
        elif isinstance(a, Scalar):
            result = a+b if op == "add" else a-b if op == "subtract" else a*b
        else:
            result = b+a if op == "add" else a-b if op == "subtract" else b*a
        parts.append(result)
    return materializer.packed_value(parts, outshape, producer=dict(node_id=node["id"], kind="packed_pointwise"))


def unary(op, x, coefficients, attrs, zero):
    if op == "negate": return -x
    if op == "square": return x*x
    if op == "power": return x**attrs["exponent"]
    if attrs["basis"] == "power": return power_polynomial(x, coefficients, zero)
    from chebyshev_lowering import balanced
    return balanced(x, coefficients, zero)


def implementation():
    result = scalar_rule.implementation()
    for path in (Path(__file__).resolve(), Path(capacity_constants.__file__).resolve()):
        result[path.name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def lower(request, *, source_kind=SOURCE_KIND):
    require(type(source_kind) is str and source_kind == SOURCE_KIND, "Rule-only source authority required")
    before = digest(request)
    q = copy.deepcopy(request)
    validate_request(q)
    require(q["layout"]["execution_abi"] in ABIS, "Mixed requires original capacity ABI")
    require(q.get("compiler_configuration", {}).get("backend") == "poseidon_cpu", "Explicit Poseidon CPU baseline")
    require(not any(k in q for k in ("construction_exercise", "construction_profile", "upstream_helpers")),
            "Free native rule baseline only")
    model = q["model"]
    checked = validate(model)
    supported = {"linear", "add", "subtract", "multiply", "negate", "square", "power", "polynomial",
                 "reshape", "flatten", "transpose", "permute", "rotate", "concat", "stack", "slice", "split", "sum", "mean"}
    require(all(node["op"] in supported for node in model["nodes"]), "Unsupported staged mixed operator")
    packed = capacity_constants.packing(model, q["layout"], capacity_constants.request_version(q))
    em = MixedEmitter(q, checked)
    materializer = Materializer(em)
    values = {name:np.asarray(value, dtype=np.float64) for name, value in model["constants"].items()}
    inputs = []
    for spec in model["inputs"]:
        bindings = [b for b in q["layout"]["inputs"] if b["name"] == spec["name"]]
        size = math.prod(spec["shape"])
        offset = 0
        for binding in bindings:
            count = min(em.period, size-offset)
            require(binding.get("offset", 0) == offset and binding.get("elements", size) == count and count > 0,
                    "Mixed input must have canonical complete chunk prefixes")
            offset += count
        require(offset == size, "Mixed complete input coverage")
        parts = [Scalar(em, b["dsl_name"]) for b in bindings]
        values[spec["name"]] = materializer.packed_value(parts, spec["shape"], producer=dict(input_name=spec["name"]))
        inputs.append(dict(name=spec["name"], shape=spec["shape"], chunks=copy.deepcopy(bindings),
                           representation="canonical_packed_input"))
    transitions = []
    views = {"reshape", "flatten", "transpose", "permute", "rotate", "concat", "stack", "slice", "split"}
    for node in model["nodes"]:
        op, attrs = node["op"], node["attrs"]
        xs = [values[ref] for ref in node["inputs"]]
        shape = checked["shapes"][node["outputs"][0]]
        if op in views:
            y = view(op, xs, attrs)
            mode = "view_index_map"
        elif op == "linear":
            require(isinstance(xs[0], TensorValue) and all(ref in model["constants"] for ref in node["inputs"][1:]),
                    "Mixed Linear requires bound original public weight/bias")
            original = materializer.chunks(xs[0], boundary=dict(node_id=node["id"], kind="linear_input"))
            result = original.linear(node, packed, shape)
            y = materializer.packed_value(result.parts, shape, producer=dict(node_id=node["id"], kind="bound_packed_linear"))
            mode = "bound_packed_linear"
        elif op in ("add", "subtract", "multiply"):
            y = packed_binary(op, xs, shape, node, packed, materializer)
            mode = "packed_pointwise"
            if y is None:
                a, b = [materializer.array(x) for x in xs]
                y = TensorValue(a+b if op == "add" else a-b if op == "subtract" else a*b)
                mode = "replicated_cells_math"
        elif op in ("sum", "mean"):
            require(isinstance(xs[0], TensorValue), "Mixed reduction requires cipher tensor")
            a = materializer.array(xs[0])
            axes = tuple(i % a.ndim for i in attrs["axes"])
            result = np.asarray(np.sum(a, axis=axes, keepdims=attrs["keepdims"]), dtype=object)
            if op == "mean": result = np.asarray(result/math.prod(a.shape[i] for i in axes), dtype=object)
            if result.ndim == 0 and shape == [1]: result = result.reshape(1)
            y = TensorValue(result)
            mode = "replicated_cells_reduction"
        else:
            require(isinstance(xs[0], TensorValue), "Mixed unary requires cipher tensor")
            coeff = xs[1] if op == "polynomial" else None
            storage = xs[0].canonical_storage()
            if storage is not None:
                parts = [unary(op, part, coeff, attrs, em.zero) for part in storage.parts]
                y = materializer.packed_value(parts, shape, producer=dict(node_id=node["id"], kind="packed_unary"))
                mode = "packed_pointwise"
            else:
                a = materializer.array(xs[0])
                result = np.empty(a.shape, dtype=object)
                for index in np.ndindex(a.shape):
                    result[index] = unary(op, a[index], coeff, attrs, em.zero)
                y = TensorValue(result)
                mode = "replicated_cells_math"
        children = y if op == "split" else [y]
        require(len(children) == len(node["outputs"]), "Mixed operator output count")
        for name, result in zip(node["outputs"], children):
            require(list(result.shape) == checked["shapes"][name], "Mixed logical shape mismatch")
            values[name] = result
        transitions.append(dict(node_id=node["id"], operator=op, representation=mode,
            outputs={name:materializer.index_map(values[name]) for name in node["outputs"]}))
    outputs, output_bindings = [], []
    for output, binding in zip(model["outputs"], q["layout"]["outputs"]):
        value = values[output["value"]]
        require(isinstance(value, TensorValue) and list(value.shape) == binding["shape"], "Mixed output logical shape")
        chunks = materializer.chunks(value, boundary=dict(output_name=output["name"], kind="physical_output"))
        require(len(chunks.parts) == len(binding["chunks"]), "Mixed output chunk count")
        offset = 0
        for part, original in zip(chunks.parts, binding["chunks"]):
            count = min(em.period, math.prod(value.shape)-offset)
            require(original["ciphertext"] == len(outputs) and original["offset"] == offset and original["elements"] == count,
                    "Mixed output immutable canonical coverage")
            outputs.append(part.name)
            output_bindings.append(dict(name=output["name"], **copy.deepcopy(original)))
            offset += count
        require(offset == math.prod(value.shape), "Mixed complete output coverage")
    require(len(outputs) == q["layout"]["output_ciphertexts"], "Mixed physical output ciphertext count")
    names = [b["dsl_name"] for b in q["layout"]["inputs"]]+["zero_ct"]
    raw_source = '@hc.func("'+','.join('c' for _ in names)+'")\ndef golden('+', '.join(names)+'):\n'
    raw_source += '\n'.join(em.lines)+'\n    return ['+', '.join(outputs)+']\n'
    raw_source = '\n'.join(body for _, body in em.capacity_helpers.values())+raw_source
    require(len(raw_source.encode()) <= 65536, "Rule source budget")
    source = capacity_lowering.compact_scalar_source(raw_source)
    output_identity = None
    if capacity_constants.request_version(q) == capacity_constants.INDEX_VERSION:
        from capacity_output_identity import distinct_outputs
        source, output_identity = distinct_outputs(source)
    static = validate_candidate(dict(schema=1, request_id=q["request_id"], hecate_source=source), q)
    fallback = scalar_rule.literal_map(source, em.fallbacks)
    require(all(row["source_locations"] for row in fallback), "Literal source occurrence missing")
    require(digest(request) == before, "Caller request changed")
    record = dict(format=FORMAT, version=VERSION, source_kind=source_kind,
        strategy="canonical_packed_linear_lazy_replicated_cells_C_order_pack",
        request_id=q["request_id"], input_request_sha256=before, restored_request_sha256=digest(q),
        model_sha256=digest(model), public_constants_sha256=digest(q["public_constants"]),
        layout_sha256=digest(q["layout"]), compiler_configuration_sha256=digest(q["compiler_configuration"]),
        public_packing_sha256=digest(packed), implementation=implementation(),
        candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
        raw_source_sha256=hashlib.sha256(raw_source.encode()).hexdigest(),
        inputs=inputs, outputs=output_bindings, storages=materializer.storages,
        transitions=transitions, materialized_cells=materializer.materializations, packs=materializer.packs,
        expression_names_graph="raw_emitter_SSA", missing_literals=fallback,
        literal_source_map_sha256=digest(fallback), public_origin_authority_loaded=em.authority_loaded,
        emitted_operations=len(em.lines), emitted_statement_bytes=em.emitted_statement_bytes,
        operations=scalar_rule.source_statistics(source),
        expanded_native_work=static["functions"]["golden"]["expanded_cost"], output_identity=output_identity,
        physical_input_chunks=len(q["layout"]["inputs"]), physical_output_chunks=len(outputs),
        supported_operators=sorted(supported), request_changed=False, model_changed=False,
        public_registry_changed=False, compiler_profile_changed=False, reference_changed=False,
        provider_candidate_rewritten=False, new_agent_generation=False, compiler_run=False,
        encrypted_execution=False, depth_is_capacity_proof=False, peak_allocation_is_known=False)
    record["binding"] = digest(record)
    return source, record
