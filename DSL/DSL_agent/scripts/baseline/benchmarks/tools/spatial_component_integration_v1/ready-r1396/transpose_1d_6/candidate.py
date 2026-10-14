@hc.func('c,c,c')
def golden(x0, x1, zero_ct):
    v3 = x0 * pv0 + x0.rotate(1) * pv1
    v8 = x0.rotate(1).rotate(2) * pv1 + x1.rotate(2) * pv0
    v9 = v3.rotate(2)
    v10 = v8.rotate(2)
    return [v3 * pv2 + v9 * pv3 + v10 * pv7 + pv10, v3 * pv4 + v9 * pv5 + (v8 * pv8 + v10 * pv3) + pv11, v9 * pv6 + (v8 * pv9 + v10 * pv5) + pv12]
