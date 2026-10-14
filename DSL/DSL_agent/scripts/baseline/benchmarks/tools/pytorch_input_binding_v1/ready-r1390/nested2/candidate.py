@hc.func('c,c,c,c,c')
def golden(x0, x1, x2, x3, zero_ct):
    v0 = x2 * pv0
    v1 = x0 * p4
    v2 = v1 * pv0
    v3 = v0 + v2
    v4 = x0 * p2
    v5 = x2 - v4
    v6 = v5.rotate(1)
    v7 = v6 * pv2
    v8 = v3 + v7
    v9 = x2 + v1
    v10 = v9.rotate(1)
    v11 = v10 * pv1
    v12 = x1 * p4
    v13 = x3 + v12
    v14 = v13.rotate(1)
    v15 = v14 * pv2
    v16 = v11 + v15
    v17 = v6 * pv4
    v18 = v16 + v17
    v19 = v14 * pv3
    v20 = v5.rotate(2)
    v21 = v20 * pv5
    v22 = v19 + v21
    v23 = x1 * p2
    v24 = x3 - v23
    v25 = v24.rotate(2)
    v26 = v25 * pv6
    v27 = v22 + v26
    return [v8, v18, v27]
