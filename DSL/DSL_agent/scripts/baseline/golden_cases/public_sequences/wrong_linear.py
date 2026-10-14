def dot(a, weight):
    value = a * weight
    value = value + value.rotate(1)
    return value + value.rotate(2)

@hc.func("c")
def golden(x):
    rows = [c1, c0][:]
    return [dot(x, row) for row in rows]
