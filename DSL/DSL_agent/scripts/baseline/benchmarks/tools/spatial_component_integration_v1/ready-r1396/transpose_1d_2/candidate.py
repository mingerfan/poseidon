@hc.func('c,c,c,c')
def golden(x0, x1, x2, zero_ct):
    v1 = x0.rotate(2) * pv0
    v6 = x0.rotate(2) * pv1 + x1.rotate(2) * pv2
    v8 = x1.rotate(2) * pv3
    v13 = x1.rotate(2) * pv1 + x2.rotate(2) * pv0
    v15 = v6.rotate(2)
    v17 = v13.rotate(2)
    return [v1 * pv4 + v1.rotate(2) * pv5 + (v6 * pv6 + v15 * pv7) + pv17, v6 * pv8 + v15 * pv9 + (v8 * pv10 + v8.rotate(2) * pv11) + (v13 * pv12 + v17 * pv13) + pv18, v13 * pv14 + v17 * pv15 + zero_ct * pv16 + pv19]
