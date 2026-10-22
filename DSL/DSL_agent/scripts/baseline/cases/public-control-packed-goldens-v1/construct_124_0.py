@hc.func('c,c')
def golden(x, zero_ct):
    v = x * p4 + p5
    v = v * x + p2
    v = v * x + p7
    low = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    high = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    a = v.rotate(2).rotate(4).rotate(8)
    b = v.rotate(1).rotate(4).rotate(8).rotate(16)
    total = a * low + b * high
    total = total + total.rotate(1)
    total = total + total.rotate(2)
    total = total + total.rotate(4)
    total = total + total.rotate(8)
    total = total + total.rotate(16)
    result = total * p6
    coverage_bias = np.polynomial.Chebyshev([0.25, 0.5]).coef[1]
    covered_out = result + coverage_bias - 0.5
    return [covered_out]
