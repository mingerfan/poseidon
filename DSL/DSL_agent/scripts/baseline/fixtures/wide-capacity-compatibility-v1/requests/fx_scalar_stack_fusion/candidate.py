@hc.func('c,c,c,c')
def golden(x, y, z, zero_ct):
    v1 = x + x.rotate(1)
    v4 = y * tail2
    v6 = v4 + v4.rotate(1)
    v11 = z * tail3
    v13 = v11 + v11.rotate(1)
    v16 = (v1 + v1.rotate(2) + (v6 + v6.rotate(2))) * p4 + (v13 + v13.rotate(2))
    return [v16 * pv0 + (v16 * v16).rotate(1).rotate(2) * pv1 + (v16 - p3).rotate(2) * pv2]
