"""Static C-order concatenation geometry; no implicit broadcasting or slot aliasing."""
import itertools
import math
from seal_artifact_gate import require


def geometry(shapes, axis, *, max_elements=8):
    require(type(max_elements) is int and max_elements in (8,16),'Invalid concat resource profile')
    require(type(shapes) in (tuple, list) and 1 <= len(shapes) <= 8,
            'Concat requires one to eight tensors')
    require(all(type(s) in (tuple, list) and 1 <= len(s) <= 4 and
                all(type(d) is int and 1 <= d <= max_elements for d in s) and math.prod(s) <= max_elements
                for s in shapes), 'Invalid concat input shape')
    rank = len(shapes[0])
    require(type(axis) is int and -rank <= axis < rank, 'Invalid concat axis')
    axis %= rank
    require(all(len(s) == rank and all(s[d] == shapes[0][d] for d in range(rank) if d != axis)
                for s in shapes), 'Concat non-axis dimensions must match')
    out = list(shapes[0])
    out[axis] = sum(s[axis] for s in shapes)
    require(math.prod(out) <= max_elements, 'Concat output element limit exceeded')
    return tuple(out), axis


def source_order(shapes, axis, *, max_elements=8):
    """Return output shape and (input index, input C-order flat index) pairs."""
    out, axis = geometry(shapes, axis,max_elements=max_elements)
    order = []
    for coord in itertools.product(*(range(d) for d in out)):
        position = coord[axis]
        branch = 0
        while position >= shapes[branch][axis]:
            position -= shapes[branch][axis]
            branch += 1
        source = list(coord)
        source[axis] = position
        flat = 0
        for index, dim in zip(source, shapes[branch]):
            flat = flat * dim + index
        order.append((branch, flat))
    return out, order


def reference(values, axis):
    """Independent recursive nested-list concat, not the emitter's flat index map."""
    def rank(value):
        return 1 + rank(value[0]) if type(value) is list else 0
    axis %= rank(values[0])
    def join(items, depth):
        if depth == 0:
            return [value for item in items for value in item]
        return [join([item[i] for item in items], depth - 1) for i in range(len(items[0]))]
    return join(values, axis)
