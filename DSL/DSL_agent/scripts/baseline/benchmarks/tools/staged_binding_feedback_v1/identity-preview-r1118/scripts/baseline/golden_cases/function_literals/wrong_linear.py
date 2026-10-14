def dot(value, weight):
    result = value * weight
    for step in [1, 2]:
        result = result + result.rotate(step)
    return result

@hc.func("c")
def golden(x):
    rows = sorted([(1, c1), (0, c0)], key=lambda row: row[0], reverse=True)
    return [(lambda pair: dot(x, pair[1]))(row) for row in rows]
