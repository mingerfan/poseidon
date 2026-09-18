def state(a):
    value = a
    def scale(weight):
        nonlocal value
        value = value * weight
    def read():
        return value
    return [scale, read]

@hc.func("c")
def golden(x):
    scale, read = state(x)
    original = read()
    scale(c0)
    return read() + original + c1
