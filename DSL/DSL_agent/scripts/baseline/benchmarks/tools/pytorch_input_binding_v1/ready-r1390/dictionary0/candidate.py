@hc.func('c,c,c,c,c')
def golden(x0, x1, x2, x3, zero_ct):
    v29 = x2 * pv0 + x2.rotate(1) * pv1 + x2.rotate(2) * pv2 + x2.rotate(1).rotate(2) * pv3 + x3.rotate(1) * pv6 + x3.rotate(2) * pv7 + pv11
    v30 = x2.rotate(2) * pv4 + x2.rotate(1).rotate(2) * pv5 + x3 * pv8 + x3.rotate(1) * pv9 + x3.rotate(1).rotate(2) * pv10 + pv12
    return [v29 * v29 + x0, v30 * v30 + x1]
