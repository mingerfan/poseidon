@hc.func('c,c,c')
def golden(x0, x1, zero_ct):
    v0 = x0 * mask0
    v2 = x0.rotate(1) * mask0
    v9 = x1 * mask0
    v11 = x1.rotate(1) * mask0
    return [(v0 + v0.rotate(1).rotate(2) + (v2.rotate(2) + v2.rotate(1))) * (v9 + v9.rotate(2) + (v11.rotate(1).rotate(2) + v11.rotate(1)))]
