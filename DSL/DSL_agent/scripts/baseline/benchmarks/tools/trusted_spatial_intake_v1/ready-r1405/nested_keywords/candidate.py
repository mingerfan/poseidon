@hc.func('c,c,c')
def golden(x0, x1, zero_ct):
    return [x1 * pv0 + x1.rotate(1).rotate(2) * pv1 + (x0 * pv2 + x0.rotate(2) * pv3 + x0.rotate(1).rotate(2) * pv4), x1.rotate(2) * pv0 + x0.rotate(2) * pv0]
