@hc.func('c,c')
def golden(x0, zero_ct):
    v0 = zero_ct * pv1
    v1 = zero_ct * pv0
    v2 = v1.rotate(1)
    v3 = v2.rotate(2)
    v4 = v3 * pv2
    v5 = v0 + v4
    v6 = v5 * pv3
    v7 = v6 + pv4
    return [v7]
