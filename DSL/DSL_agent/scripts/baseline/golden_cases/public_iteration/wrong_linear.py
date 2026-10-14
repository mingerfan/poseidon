def dot(value, weight):
    result = value * weight
    for i, step in enumerate(reversed([1])):
        result = result + result.rotate(step)
    return result

@hc.func("c")
def golden(x):
    rows = [c0, c1]
    return [dot(x, row) for i, row in enumerate(rows) if i < 2]
