"""Pure costs for existing public capacity layouts.

Depth counts serial ciphertext-by-plaintext multiplications in the public
construction, including masks. It is not a SEAL/Poseidon level estimate or a
compiler feasibility claim. All-zero/all-one simplifications may reduce actual
work and depth; the reported depth deliberately keeps their structural upper
bound. No DSL, backend, parameters, reference values or runtime are consulted.
"""
import copy
import math
from unified_capacity_tiling import template_cost, template_vectors, range_vector

FORMAT = "poseidon-capacity-layout-costs-v1"
MAX_PERIOD = 4096
MAX_BLOCKS = 1024
MAX_TEMPLATE_VECTORS = 1024
DEFAULT_MAX_WORK = 1023
DEFAULT_MAX_CONSTANTS = 256


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, label, *, lower=0, upper=None):
    _require(type(value) is int and value >= lower
             and (upper is None or value <= upper), label)
    return value


def _period(period):
    _integer(period, "Cost period", lower=4, upper=MAX_PERIOD)
    _require(period & (period - 1) == 0, "Cost power-of-two period")
    return period


def _vector(value, period):
    _require(type(value) in (list, tuple) and len(value) == period,
             "Cost public vector shape")
    _require(all(type(v) in (int, float) and math.isfinite(v) for v in value),
             "Cost finite public vector")
    return tuple(value)


def _template(t, period):
    _require(type(t) is dict and {"stride", "expand", "compress"} <= set(t),
             "Cost template fields")
    stride = _integer(t["stride"], "Cost template stride", lower=1, upper=period)
    _require(stride & (stride - 1) == 0, "Cost power-of-two stride")
    row_sum = "row_weights" in t
    expected = {"stride", "expand", "compress",
                *({"row_weights", "row_mask"} if row_sum else {"weights"})}
    _require(set(t) == expected, "Cost template fields")
    for role in ("expand", "compress"):
        stages = t[role]
        _require(type(stages) is list and len(stages) <= period.bit_length(),
                 "Cost permutation stage bound")
        for stage in stages:
            _require(type(stage) is dict and set(stage) == {"shift", "fixed", "moving"},
                     "Cost permutation stage")
            _integer(stage["shift"], "Cost stage rotation", upper=period - 1)
            for side in ("fixed", "moving"):
                vector = _vector(stage[side], period)
                _require(all(v in (0, 1) for v in vector), "Cost permutation mask")
    weights = t["row_weights"] if row_sum else t["weights"]
    _require(type(weights) is dict and len(weights) <= MAX_TEMPLATE_VECTORS,
             "Cost weight vector bound")
    if row_sum:
        _require(weights and set(weights) == set(range(len(weights))),
                 "Cost row output indices")
    for key, vector in weights.items():
        _integer(key, "Cost weight index", upper=(stride - 1 if row_sum else period - 1))
        _vector(vector, period)
    if row_sum:
        _require(all(v in (0, 1) for v in _vector(t["row_mask"], period)),
                 "Cost row mask")
    return row_sum


def _record(name, depth, work, count, *, stages=None, block_depths=None):
    _require(type(name) is str and 1 <= len(name) <= 128, "Cost candidate name")
    for value, label in ((depth, "Cost plaintext depth"), (work, "Cost work"),
                         (count, "Cost constants")):
        if value is not None:
            _integer(value, label)
    result = dict(
        format=FORMAT, name=name, plaintext_depth_upper_bound=depth,
        estimated_work=work, constant_count=count,
        certainty=dict(plaintext_depth="conservative_upper_bound" if depth is not None else "unknown",
                       work="estimate" if work is not None else "unknown",
                       constants="exact_count" if count is not None else "unknown"),
        compiler_feasibility="unknown", execution_feasibility="unknown",
        zero_one_elision="not_discounted_in_depth_upper_bound",
    )
    if stages is not None:
        result["depth_stages"] = copy.deepcopy(stages)
    if block_depths is not None:
        result["block_depths"] = copy.deepcopy(block_depths)
    return result


def candidate_costs(name, plaintext_depth_upper_bound, estimated_work, constant_count):
    """Describe a known construction or unknown estimate without certifying it.

    Direct diagonal and direct row-sum constructions use depth bounds one and
    two respectively. Callers obtain their work and constant counts from the
    existing public layout planner. None means that metric is not known.
    """
    return _record(name, plaintext_depth_upper_bound, estimated_work, constant_count)


def template_costs(t, period):
    """Cost one existing template; reuse the existing public work estimator."""
    period = _period(period)
    row_sum = _template(t, period)
    vectors = list(template_vectors(t))
    _require(len(vectors) <= MAX_TEMPLATE_VECTORS, "Cost template vector bound")
    unique = {_vector(vector, period) for *_, vector in vectors}
    stages = dict(expand=len(t["expand"]), weight=1,
                  row_mask=int(row_sum), compress=len(t["compress"]))
    return _record("tiled_row_sum" if row_sum else "tiled_diagonal",
                   sum(stages.values()), template_cost(t, period), len(unique),
                   stages=stages)


def tiled_costs(plan, templates, period, *, existing_vectors=()):
    """Aggregate independent blocks by maximum depth, and cumulative work.

    Masks add one serial multiplication only when a block gathers/scatters more
    than one chunk. Vectors are deduplicated together with the existing public
    registry; shared vector contents consume one constant entry. Work includes
    helper calls and the existing conservative per-call registry charge.
    """
    period = _period(period)
    _require(type(plan) is dict and type(templates) is dict and templates,
             "Cost tiled plan")
    blocks = plan.get("blocks")
    _require(type(blocks) is list and 1 <= len(blocks) <= MAX_BLOCKS,
             "Cost tiled block bound")
    _require(type(existing_vectors) in (list, tuple, set, frozenset, dict),
             "Cost existing public vectors")
    supplied = existing_vectors.values() if type(existing_vectors) is dict else existing_vectors
    _require(len(existing_vectors) <= DEFAULT_MAX_CONSTANTS,
             "Cost existing registry bound")
    unique = {_vector(v, period) for v in supplied}
    local = {}
    for count, t in templates.items():
        _integer(count, "Cost tile batch count", lower=1, upper=period)
        local[count] = template_costs(t, period)
        _require(count * t["stride"] <= period, "Cost tile capacity")
        unique.update(_vector(v, period) for *_, v in template_vectors(t))
    depths, work, used = [], 0, set()
    for block in blocks:
        _require(type(block) is dict and
                 {"batch_offset", "batches", "gather", "scatter"} <= set(block),
                 "Cost tiled block fields")
        _integer(block["batch_offset"], "Cost batch offset")
        count = _integer(block["batches"], "Cost block batch count", lower=1)
        _require(count in local, "Cost missing tile template")
        used.add(count)
        stages = dict(local[count]["depth_stages"])
        stages.update(gather=0, scatter=0)
        work += local[count]["estimated_work"] + 1
        for role in ("gather", "scatter"):
            spans = block[role]
            _require(type(spans) is list and 1 <= len(spans) <= period,
                     "Cost block spans")
            for span in spans:
                _require(type(span) is dict and {"chunk", "lo", "hi", "step"} <= set(span),
                         "Cost block span fields")
                _integer(span["chunk"], "Cost span chunk")
                lo = _integer(span["lo"], "Cost span lower", upper=period - 1)
                hi = _integer(span["hi"], "Cost span upper", lower=1, upper=period)
                _require(lo < hi, "Cost span interval")
                work += _integer(span["step"], "Cost span rotation", upper=period - 1).bit_count()
            if len(spans) > 1:
                stages[role] = 1
                work += len(spans)
                unique.update(tuple(range_vector(s["lo"], s["hi"], period)) for s in spans)
        work += len(block["gather"]) - 1 + len(block["scatter"])
        depths.append(dict(batch_offset=block["batch_offset"], batches=count,
                           plaintext_depth_upper_bound=sum(stages.values()),
                           depth_stages=stages))
    _require(used == set(templates), "Cost unused tile template")
    work += (len(blocks) + 1) * len(unique)
    methods = {local[c]["name"] for c in used}
    _require(len(methods) == 1, "Cost mixed tile methods")
    return _record(next(iter(methods)),
                   max(row["plaintext_depth_upper_bound"] for row in depths),
                   work, len(unique), block_depths=depths)


def choose_candidate(candidates, *, max_work=DEFAULT_MAX_WORK,
                     max_constants=DEFAULT_MAX_CONSTANTS):
    """Return a detached best planning candidate, or None.

    Budget bounds are inclusive: max_work=1023 preserves an existing cost<1024
    policy. Missing metrics are not selectable. Ranking never upgrades unknown
    compiler/execution feasibility to a successful acceptance status.
    """
    _integer(max_work, "Cost selection work bound")
    _integer(max_constants, "Cost selection registry bound",
             upper=DEFAULT_MAX_CONSTANTS)
    _require(type(candidates) in (list, tuple) and len(candidates) <= MAX_BLOCKS,
             "Cost candidate list")
    valid, names = [], set()
    for given in candidates:
        _require(type(given) is dict and given.get("format") == FORMAT,
                 "Cost candidate format")
        rebuilt = _record(given.get("name"), given.get("plaintext_depth_upper_bound"),
                          given.get("estimated_work"), given.get("constant_count"))
        for key in rebuilt:
            _require(given.get(key) == rebuilt[key], "Cost candidate identity")
        _require(given["name"] not in names, "Cost duplicate candidate")
        names.add(given["name"])
        depth, work, count = (given[k] for k in (
            "plaintext_depth_upper_bound", "estimated_work", "constant_count"))
        if None in (depth, work, count):
            continue
        if work <= max_work and count <= max_constants:
            valid.append(given)
    if not valid:
        return None
    return copy.deepcopy(min(valid, key=lambda c: (
        c["plaintext_depth_upper_bound"], c["estimated_work"],
        c["constant_count"], c["name"])))
