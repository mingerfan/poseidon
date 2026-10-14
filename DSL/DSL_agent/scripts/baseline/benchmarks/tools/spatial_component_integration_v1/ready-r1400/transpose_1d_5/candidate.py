@hc.func('c,c')
def golden(x0, zero_ct):
    return [x0 * pv0 + x0.rotate(2) * pv1 + x0.rotate(1).rotate(2) * pv2 + pv8, x0.rotate(1) * pv3 + x0.rotate(2) * pv4 + pv9, x0 * pv5 + x0.rotate(1) * pv6 + x0.rotate(1).rotate(2) * pv7 + pv10]
