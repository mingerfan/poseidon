@hc.func('c,c')
def golden(x, zero_ct):
    v = x + p4
    active = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    total = v * active
    total = total + total.rotate(1)
    total = total + total.rotate(2)
    total = total + total.rotate(4)
    total = total + total.rotate(8)
    total = total + total.rotate(16)
    result = total * p3
    coverage_bias = np.array([0.5])[0]
    covered_out = result + coverage_bias - 0.5
    return [covered_out]
