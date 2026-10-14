@hc.func('c,c')
def golden(x0, zero_ct):
    v0 = zero_ct * pv1
    v1 = v0.rotate(1)
    v2 = v1.rotate(2)
    v3 = v2 * pv5
    v4 = v0 + v3
    v5 = zero_ct.rotate(1)
    v6 = v5 * pv1
    v7 = v6.rotate(2)
    v8 = v4 + v7
    v9 = v1 * pv5
    v10 = v8 + v9
    v11 = v10 * pv6
    v12 = v11 + pv7
    return [v12]
