@hc.func("c,p")
def scale(value, weight):
    value *= weight
    return value

@hc.func("c")
def golden(x):
    value = scale(*[x, c0])
    for i in range(2):
        value += scale(x, c0)
    value += c1
    return value
