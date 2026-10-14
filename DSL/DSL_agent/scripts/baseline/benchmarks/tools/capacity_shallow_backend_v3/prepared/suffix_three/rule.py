@hc.func('c,c')
def golden(x0, zero_ct):
    v0 = x0 * x0
    v4 = v0 + v0.rotate(1) + v0.rotate(2)
    v5 = v4.rotate(2)
    v6 = v5.rotate(2)
    return [v4 * pv0 + v5 * pv1 + (v6 * pv2 + v6.rotate(2) * pv3)]
