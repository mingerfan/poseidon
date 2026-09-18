@hc.func("c")
def identity(value):
    return value

@hc.func("c")
def golden(x):
    items = np.array([c0, x], dtype=object)
    view = items[:1]
    view += x
    return identity(items[0]) + x * c0 + c1 - c0
