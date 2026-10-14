@hc.func('c,c')
def golden(x0, zero_ct):
    v0 = x0 * x0
    v1 = v0.rotate(1)
    v2 = v1.rotate(1).rotate(2).rotate(8)
    v3 = v2.rotate(1)
    v4 = v3.rotate(1)
    return [v0 * pv0 + v1 * pv1 + v2 * pv2 + v3 * pv3 + v4 * pv4 + v4.rotate(1) * pv5 + pv6]
