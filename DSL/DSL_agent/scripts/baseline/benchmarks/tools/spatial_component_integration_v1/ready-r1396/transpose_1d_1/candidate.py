@hc.func('c,c,c,c')
def golden(x0, x1, x2, zero_ct):
    v4 = x0.rotate(2) * pv0 + x0.rotate(1).rotate(2) * pv1
    v6 = x0.rotate(1) * pv1
    v11 = x0.rotate(1).rotate(2) * pv2 + x1.rotate(2) * pv3
    v16 = x1.rotate(1) * pv2 + x1.rotate(1).rotate(2) * pv0
    v21 = x1.rotate(2) * pv1 + x2.rotate(1) * pv0
    v23 = x2.rotate(1).rotate(2) * pv3
    v28 = x2.rotate(1) * pv3 + x2.rotate(2) * pv2
    v30 = v6.rotate(1)
    v32 = v11.rotate(1).rotate(2)
    v33 = v16.rotate(1)
    v35 = v21.rotate(1)
    v36 = v23.rotate(1)
    v37 = v36.rotate(2)
    return [v4 * pv4 + v4.rotate(1) * pv5 + (v30 * pv6 + v30.rotate(2) * pv7) + (v11 * pv10 + v32 * pv11) + pv23, v6 * pv8 + v30 * pv9 + v32 * pv7 + (v16 * pv12 + (v33 * pv13 + v33.rotate(2) * pv14)) + v35 * pv15 + (v23 * pv18 + v37 * pv19) + pv24, v21 * pv16 + v35 * pv17 + (v36 * pv15 + v37 * pv20) + (v28 * pv21 + v28.rotate(1).rotate(2) * pv22) + pv25]
