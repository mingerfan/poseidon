@hc.func("c,c,c")
def golden(x, y, zero_ct):
    coef = {'a00': p2, 'a01': p4, 'a02': p6,
            'a10': p7, 'a11': p3, 'a12': p5}
    vals = list(coef.values())
    a00 = vals[0]
    a01 = vals[1]
    a02 = vals[2]
    a10 = vals[3]
    a11 = vals[4]
    a12 = vals[5]

    m0 = np.array([1.0, 0.0, 0.0, 0.0])
    m1 = np.array([0.0, 1.0, 0.0, 0.0])

    d = x - y

    s0 = d * (m0 * a00) + d.rotate(1) * (m0 * a01) + d.rotate(2) * (m0 * a02)
    s1 = (d.rotate(2).rotate(1)) * (m1 * a10) + d * (m1 * a11)

    res = zero_ct + s0 + s1
    return [res]
