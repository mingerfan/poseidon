@hc.func('c,c,c')
def golden(x0, x1, zero_ct):
    return [x0 * pv0 + x0.rotate(1).rotate(2) * pv1, x0.rotate(2) * pv2 + x0.rotate(1).rotate(2) * pv3, x0.rotate(1) * pv4 + x0.rotate(2) * pv5 + x1.rotate(1) * pv7, x0 * pv6 + x1 * pv8 + x1.rotate(1).rotate(2) * pv9, x1 * pv10 + x1.rotate(2) * pv6 + x1.rotate(1).rotate(2) * pv11]
