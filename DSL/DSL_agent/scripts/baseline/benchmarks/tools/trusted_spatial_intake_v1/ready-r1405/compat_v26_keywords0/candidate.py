@hc.func('c,c,c,c,c')
def golden(x0, x1, x2, x3, zero_ct):
    return [x2 * pv0 + x2.rotate(1) * pv1 + x2.rotate(2) * pv2 + x2.rotate(1).rotate(2) * pv3 + x3.rotate(1) * pv6 + x3.rotate(2) * pv7 + x0 * p3, x2.rotate(2) * pv4 + x2.rotate(1).rotate(2) * pv5 + x3 * pv8 + x3.rotate(1) * pv9 + x3.rotate(1).rotate(2) * pv10 + x1 * p3]
