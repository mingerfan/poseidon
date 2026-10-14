def finish(a, b):
    return a + b + c1

@hc.func("c")
def golden(x):
    items = [x, x * c0]
    items[:] = [x, items[1]]
    return finish(items[0], x)
