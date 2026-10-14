@hc.func('c,c')
def golden(x0, zero_ct):
    return [x0 * pv0 + x0.rotate(1) * pv1 + x0.rotate(2) * pv2 + x0.rotate(1).rotate(2) * pv3 + pv4]
