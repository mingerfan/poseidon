@hc.func('c,c,c,c')
def golden(x0, x1, x2, zero_ct):
    return [x0 * pv0 + x0.rotate(1).rotate(2) * pv1, x0.rotate(1) * pv2 + x0.rotate(2) * pv3 + x1.rotate(1) * pv4, x1 * pv5 + x1.rotate(1) * pv0 + x1.rotate(1).rotate(2) * pv4, x1.rotate(1).rotate(2) * pv0 + x2.rotate(2) * pv6 + x2.rotate(1).rotate(2) * pv7, x2 * pv4 + x2.rotate(1) * pv8]
