@hc.func('c,c,c')
def golden(x0, x1, zero_ct):
    v4 = x0.rotate(2) * pv0 + x0.rotate(1).rotate(2) * pv1
    v6 = x0.rotate(1) * pv1
    v11 = x0.rotate(2) * pv1 + x1.rotate(1) * pv0
    v12 = x1 * pv1
    v13 = v4.rotate(1)
    v14 = v13.rotate(1)
    v15 = v14.rotate(1)
    v16 = v6.rotate(1)
    v17 = v16.rotate(1)
    v19 = v11.rotate(1)
    v20 = v19.rotate(1)
    v21 = v20.rotate(1)
    v22 = v12.rotate(1)
    v23 = v22.rotate(1)
    v46 = v4 * pv2 + (v13 * pv3 + v14 * pv4) + (v16 * pv10 + v17 * pv11) + (v11 * pv16 + (v19 * pv17 + v20 * pv18)) + (v22 * pv24 + v23 * pv25) + pv30
    v47 = v4 * pv5 + (v14 * pv6 + v15 * pv7) + (v6 * pv12 + (v16 * pv13 + v17 * pv4)) + (v11 * pv19 + (v20 * pv20 + v21 * pv21)) + (v12 * pv26 + (v22 * pv27 + v23 * pv18)) + pv31
    v48 = v14 * pv8 + v15 * pv9 + (v6 * pv14 + (v17 * pv6 + v17.rotate(1) * pv15)) + (v20 * pv22 + v21 * pv23) + (v12 * pv28 + (v23 * pv20 + v23.rotate(1) * pv29)) + pv32
    return [v46, v47, v48, v46 * v46, v47 * v47, v48 * v48]
