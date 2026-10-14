@hc.func('c,c')
def golden(x0, zero_ct):
    v0 = x0 * pv0
    v1 = v0 * pv4
    v2 = x0.rotate(2)
    v3 = v2 * pv1
    v4 = v3 * pv4
    v5 = v1 + v4
    v6 = x0.rotate(1)
    v7 = v6.rotate(2)
    v8 = v7 * pv2
    v9 = v8 * pv4
    v10 = v5 + v9
    v11 = v0 + v3
    v12 = v11 + v8
    v13 = v12 * v12
    v14 = v2 * pv3
    v15 = v14 * v14
    return [v10, zero_ct, v13, v15]
