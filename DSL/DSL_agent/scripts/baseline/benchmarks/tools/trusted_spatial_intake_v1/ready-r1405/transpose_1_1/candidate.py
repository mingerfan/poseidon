@hc.func('c,c,c')
def golden(x0, x1, zero_ct):
    return [x0 * pv0 + x0.rotate(2) * pv1 + x0.rotate(1).rotate(2) * pv2 + pv8, x0 * pv3 + x0.rotate(1) * pv4 + x0.rotate(2) * pv0 + pv9, x0.rotate(1).rotate(2) * pv5 + x1.rotate(2) * pv6 + x1.rotate(1).rotate(2) * pv7 + pv10, x1.rotate(1) * pv5 + pv11]
