@hc.func('c,c')
def golden(x0, zero_ct):
    v6 = x0 * pv0 + x0.rotate(2) * pv1 + x0.rotate(1).rotate(2) * pv2
    v8 = x0.rotate(2) * pv3
    return [v6, v8, v6 + p3, v8 + p3]
