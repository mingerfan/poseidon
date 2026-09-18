@hc.func("c")
def golden(x):
    product0 = x * c0
    half0 = product0 + product0.rotate(2)
    sum0 = half0 + half0.rotate(1)
    value0 = sum0 + c1
    return value0
