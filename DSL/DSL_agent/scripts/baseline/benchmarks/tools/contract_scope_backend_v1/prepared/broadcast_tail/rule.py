@hc.func('c,c,c,c')
def golden(x0, x1, x2, zero_ct):
    v0 = x2 * mask0
    v2 = x2.rotate(1) * mask0
    v4 = x2.rotate(2) * mask0
    v6 = v0 + v0.rotate(1).rotate(2).rotate(4)
    v20 = v4 + v4.rotate(1).rotate(2).rotate(4)
    v27 = x0 + (v6 + v6.rotate(2).rotate(4) + v0.rotate(4) + (v2.rotate(1).rotate(2) + v2.rotate(2) + v2.rotate(1)))
    v28 = x1 + (v2 + v2.rotate(1).rotate(2).rotate(4) + (v20 + v20.rotate(2).rotate(4) + v4.rotate(4)).rotate(2).rotate(4))
    return [v27 * v27, v28 * v28]
