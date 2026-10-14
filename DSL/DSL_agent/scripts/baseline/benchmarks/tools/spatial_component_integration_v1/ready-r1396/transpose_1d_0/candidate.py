@hc.func('c,c')
def golden(x0, zero_ct):
    v1 = x0.rotate(2) * pv0
    v3 = x0.rotate(2) * pv1
    v4 = v1.rotate(1)
    v5 = v4.rotate(1)
    v6 = v5.rotate(1)
    v7 = v3.rotate(1)
    v8 = v7.rotate(1)
    return [v1 * pv2 + (v4 * pv3 + v5 * pv4) + (v7 * pv10 + v8 * pv11), v1 * pv5 + (v5 * pv6 + v6 * pv7) + (v3 * pv12 + (v7 * pv13 + v8 * pv4)), v5 * pv8 + v6 * pv9 + (v3 * pv14 + (v8 * pv6 + v8.rotate(1) * pv15))]
