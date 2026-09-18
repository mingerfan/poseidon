@hc.func("c")
def golden(x):
    product0 = x * c0
    half0 = product0 + product0.rotate(2)
    sum0 = half0 + half0.rotate(1)
    product1 = x * c1
    half1 = product1 + product1.rotate(2)
    sum1 = half1 + half1.rotate(1)
    return [sum1, sum1]
