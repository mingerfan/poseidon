@hc.func("c")
def identity(value):
    return value

@hc.func("p")
def public_identity(value):
    return value

@hc.func("c")
def golden(x):
    items = np.array([[c0, c0], [c0, c0]], dtype=object)
    transposed = items.T
    copied = np.array(transposed, dtype=object)
    flattened = copied.reshape(4)
    flattened += x
    return identity(flattened[0]) + x * c0 + c1 - public_identity(copied[0, 0])
