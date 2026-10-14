@hc.func('c,c')
def golden(x0, zero_ct):
    v4 = x0.rotate(2) * pv0 + x0.rotate(1).rotate(2) * pv1
    v6 = x0.rotate(1) * pv1
    v7 = v4.rotate(1)
    v8 = v7.rotate(1)
    v9 = v8.rotate(1)
    v10 = v6.rotate(1)
    v11 = v10.rotate(1)
    return [v4 * pv2 + (v7 * pv3 + v8 * pv4) + (v10 * pv10 + v11 * pv11) + pv16, v4 * pv5 + (v8 * pv6 + v9 * pv7) + (v6 * pv12 + (v10 * pv13 + v11 * pv4)) + pv17, v8 * pv8 + v9 * pv9 + (v6 * pv14 + (v11 * pv6 + v11.rotate(1) * pv15)) + pv18]
