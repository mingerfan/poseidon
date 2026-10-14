@hc.func('c,c')
def golden(x0, zero_ct):
    return [x0 * pv0 + x0.rotate(2) * pv1 + x0.rotate(1).rotate(2) * pv2, x0.rotate(1) * pv3 + x0.rotate(2) * pv4 + x0.rotate(1).rotate(2) * pv5, x0 * pv6 + x0.rotate(1) * pv7 + x0.rotate(2) * pv8]
