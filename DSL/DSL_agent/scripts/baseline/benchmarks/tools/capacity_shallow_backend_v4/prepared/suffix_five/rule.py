@hc.func('c,c')
def golden(x0, zero_ct):
    v0 = x0 * x0
    v2 = v0 + v0.rotate(1)
    v6 = v2 + v2.rotate(2) + v0.rotate(4)
    v7 = v6.rotate(4)
    return [v6 * pv0 + (v7 * pv1 + v7.rotate(4) * pv2)]
