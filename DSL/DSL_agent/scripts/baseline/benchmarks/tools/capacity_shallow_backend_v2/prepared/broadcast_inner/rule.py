@hc.func('c,c,c,c')
def golden(x0, x1, x2, zero_ct):
    v0 = x2 * mask0
    v2 = x2.rotate(1) * mask0
    v4 = x2.rotate(2) * mask0
    v6 = v0 + v0.rotate(1).rotate(2).rotate(4).rotate(8)
    v18 = v2 + v2.rotate(1).rotate(2).rotate(4).rotate(8)
    v24 = v2 + v2.rotate(1).rotate(2).rotate(4).rotate(8)
    v28 = v4 + v4.rotate(1).rotate(2).rotate(4).rotate(8)
    v34 = v4 + v4.rotate(1).rotate(2).rotate(4).rotate(8)
    v39 = x0 + (v6 + v6.rotate(2).rotate(4).rotate(8) + (v0.rotate(4) + v0.rotate(1).rotate(2) + v0.rotate(2) + v0.rotate(1)) + (v18 + v18.rotate(2).rotate(4).rotate(8)).rotate(4).rotate(8) + (v28 + v28.rotate(2).rotate(4).rotate(8)).rotate(8))
    v40 = x1 + (v24 + v24.rotate(2).rotate(4).rotate(8) + (v34 + v34.rotate(2).rotate(4).rotate(8)).rotate(4).rotate(8))
    return [v39 * v39, v40 * v40]
