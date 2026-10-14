@hc.func("c")
def pack(value):
    return np.array([c0], dtype=object)

@hc.func("c")
def identity(value):
    return value

@hc.func("p")
def public_identity(value):
    return value

@hc.func("c")
def golden(x):
    first = pack(x)
    second = pack(x)
    first += x
    return identity(first[0]) + x * c0 + c1 - public_identity(second[0])
