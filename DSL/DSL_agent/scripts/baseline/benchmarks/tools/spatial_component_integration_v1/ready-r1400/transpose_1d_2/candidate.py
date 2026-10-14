@hc.func('c,c,c,c')
def golden(x0, x1, x2, zero_ct):
    return [x0 * pv0 + x0.rotate(2) * pv1 + x1.rotate(2) * pv3 + pv9, x0.rotate(2) * pv2 + x1 * pv4 + x1.rotate(2) * pv5 + x2.rotate(2) * pv7 + pv10, x1.rotate(2) * pv6 + x2 * pv8 + pv11]
