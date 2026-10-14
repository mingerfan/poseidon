@hc.func('c,c,c')
def golden(x0, x1, zero_ct):
    v1 = x0 + x0.rotate(1)
    v5 = x1 + x1.rotate(1)
    return [x0, x1, (v1 + v1.rotate(2)) * pv1 + (v5 + v5.rotate(2)).rotate(1).rotate(2) * pv2]
