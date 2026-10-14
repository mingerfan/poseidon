@hc.func('c,c')
def golden(x0, zero_ct):
    v1 = x0.rotate(2) * pv0
    v3 = x0.rotate(2) * pv1
    v4 = v1.rotate(2)
    v5 = v3.rotate(2)
    return [v1 * pv2 + v4 * pv3 + v5 * pv7, v1 * pv4 + v4 * pv5 + (v3 * pv8 + v5 * pv3), v4 * pv6 + (v3 * pv9 + v5 * pv5)]
