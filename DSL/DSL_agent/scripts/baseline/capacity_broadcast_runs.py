"""Bounded replication of a masked scalar into one ciphertext chunk.

The caller supplies a value whose only nonzero logical lane is lane zero.
Offsets are physical destinations inside this one chunk, never logical indices
across chunks. Rotations and addition preserve that placement exactly.
"""
VERSION = "poseidon-capacity-broadcast-runs-v1"
MAX_PERIOD = 4096
MAX_WORK = 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class _Recipe:
    def __init__(self, period):
        self.period = period
        self.actions = []
        self.rotations = 0
        self.additions = 0

    def rotate(self, value, shift):
        shift %= self.period
        if not shift:
            return value
        self.actions.append(("rotate", value, shift))
        self.rotations += shift.bit_count()
        return len(self.actions)

    def add(self, left, right):
        self.actions.append(("add", left, right))
        self.additions += 1
        return len(self.actions)

    @property
    def work(self):
        return self.rotations + self.additions


def _individual(recipe, offsets):
    result = None
    for offset in offsets:
        term = recipe.rotate(0, -offset)
        result = term if result is None else recipe.add(result, term)
    return result


def _replicate(recipe, start, stride, count):
    powers = [0]
    for bit in range(1, count.bit_length()):
        shifted = recipe.rotate(powers[-1], -(stride << (bit - 1)))
        powers.append(recipe.add(powers[-1], shifted))
    result = None
    consumed = 0
    for bit in reversed(range(count.bit_length())):
        if count & (1 << bit):
            term = recipe.rotate(powers[bit], -stride * consumed)
            result = term if result is None else recipe.add(result, term)
            consumed += 1 << bit
    return recipe.rotate(result, -start)


def _runs(offsets):
    index = 0
    while index < len(offsets):
        end = index + 1
        stride = 0
        if end < len(offsets):
            stride = offsets[end] - offsets[index]
            end += 1
            while end < len(offsets) and offsets[end] - offsets[end - 1] == stride:
                end += 1
        yield offsets[index:end], stride
        index = end


def broadcast_run_plan(offsets, period, *, max_work=MAX_WORK):
    """Return a deterministic placement recipe and primitive-work accounting.

    This counts generated additions and power-of-two rotation primitives, not
    runtime cost, compiler level consumption, or the whole candidate's budget.
    The unchanged candidate checks still charge all combined operations.
    """
    _require(type(period) is int and 4 <= period <= MAX_PERIOD
             and not period & (period - 1), "Broadcast period")
    _require(type(max_work) is int and 1 <= max_work <= MAX_WORK,
             "Broadcast work bound")
    _require(type(offsets) in (list, tuple) and 1 <= len(offsets) <= period,
             "Broadcast destination count")
    _require(all(type(offset) is int and 0 <= offset < period for offset in offsets),
             "Broadcast destination range")
    _require(all(a < b for a, b in zip(offsets, offsets[1:])),
             "Broadcast destinations must be strictly increasing")
    offsets = tuple(offsets)
    recipe = _Recipe(period)
    result = None
    records = []
    for members, stride in _runs(offsets):
        trial = _Recipe(period)
        _replicate(trial, members[0], stride, len(members))
        individual = _Recipe(period)
        _individual(individual, members)
        compact = trial.work < individual.work
        term = (_replicate(recipe, members[0], stride, len(members))
                if compact else _individual(recipe, members))
        result = term if result is None else recipe.add(result, term)
        records.append(dict(start=members[0], stride=stride, count=len(members),
                            mode="binary" if compact else "individual"))
        _require(recipe.work <= max_work, "Broadcast primitive work budget")
    individual = _Recipe(period)
    _individual(individual, offsets)
    return dict(version=VERSION, period=period, destinations=len(offsets), runs=records,
                actions=recipe.actions, result=result,
                primitive_rotations=recipe.rotations, additions=recipe.additions,
                work=recipe.work, individual_work=individual.work)


def emit_broadcast_runs(em, value, offsets, *, max_work=MAX_WORK):
    """Place one masked lane-zero expression at the given same-chunk offsets."""
    plan = broadcast_run_plan(offsets, em.period, max_work=max_work)
    values = [value]
    for kind, left, right in plan["actions"]:
        result = (em.rotate(values[left], right) if kind == "rotate"
                  else values[left] + values[right])
        values.append(result)
    return values[plan["result"]]
