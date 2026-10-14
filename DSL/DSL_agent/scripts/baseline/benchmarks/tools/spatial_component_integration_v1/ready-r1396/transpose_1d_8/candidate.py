@hc.func('c,c,c')
def golden(x0, x1, zero_ct):
    v3 = x0 * pv0 + x0.rotate(1).rotate(2) * pv1
    v8 = x0.rotate(1) * pv1 + x0.rotate(2) * pv0
    v12 = x1 * pv0 + x1.rotate(1).rotate(2) * pv1
    v13 = v3.rotate(1)
    v14 = v13.rotate(2)
    v15 = v8.rotate(1)
    v16 = v15.rotate(2)
    v17 = v12.rotate(1)
    return [v3 * pv2 + v13 * pv3 + v15 * pv7, v3 * pv4 + v14 * pv5 + (v8 * pv8 + v15 * pv9), v14 * pv6 + (v8 * pv10 + (v15 * pv11 + v16 * pv12)) + v17 * pv16, v8 * pv13 + v16 * pv14 + (v12 * pv17 + v17 * pv18), v16 * pv15 + (v12 * pv19 + v17.rotate(2) * pv20)]
