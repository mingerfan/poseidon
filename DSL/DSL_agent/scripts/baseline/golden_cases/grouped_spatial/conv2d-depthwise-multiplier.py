@hc.func("c")
def golden(x):
    product0 = x * c0
    half0 = product0 + product0.rotate(2)
    sum0 = half0 + half0.rotate(1)
    value0 = sum0 + c1
    product1 = x * c2
    half1 = product1 + product1.rotate(2)
    sum1 = half1 + half1.rotate(1)
    value1 = sum1 + c3
    product2 = x * c4
    half2 = product2 + product2.rotate(2)
    sum2 = half2 + half2.rotate(1)
    value2 = sum2 + c5
    product3 = x * c6
    half3 = product3 + product3.rotate(2)
    sum3 = half3 + half3.rotate(1)
    value3 = sum3 + c7
    return [value0, value1, value2, value3]
