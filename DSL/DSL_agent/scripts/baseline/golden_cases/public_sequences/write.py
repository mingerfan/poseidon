def finish(a, b):
    return a + b + c1

@hc.func("c")
def golden(x):
    items = [x, x * c0]
    items[::-1] = [x, items[1]]
    return finish(items[0], x)
