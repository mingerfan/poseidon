@hc.func('c,c,c')
def golden(x0, x1, zero_ct):
    v0 = x0 * x0
    v2 = v0.rotate(1)
    v3 = v2.rotate(1)
    v4 = v3.rotate(1)
    v6 = (x1 * x1).rotate(2)
    v7 = v6.rotate(1)
    v8 = v7.rotate(1)
    v9 = v8.rotate(1)
    return [v0 * pv0 + v2 * pv1 + v3 * pv2 + v4 * pv3 + v4.rotate(4) * pv4 + v6 * pv5 + v7 * pv6 + v8 * pv7 + v9 * pv8 + pv12, v7 * pv4 + v8 * pv9 + v9 * pv10 + v9.rotate(1) * pv11 + pv13]
