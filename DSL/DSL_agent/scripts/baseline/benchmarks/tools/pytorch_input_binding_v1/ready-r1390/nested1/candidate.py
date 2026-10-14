@hc.func('c,c,c')
def golden(x0, x1, zero_ct):
    v0 = x1 * pv0
    v1 = x0 * mask0
    v2 = x0.rotate(1)
    v3 = v2 * mask0
    v4 = v3.rotate(1)
    v5 = v4.rotate(2)
    v6 = v5 * pv0
    v7 = v1 + v6
    v8 = v1.rotate(2)
    v9 = v8 * pv0
    v10 = v7 + v9
    v11 = v4 * pv0
    v12 = v10 + v11
    v13 = v0 + v12
    v14 = v1 + v5
    v15 = v14 + v8
    v16 = v15 + v4
    v17 = x1 + v16
    v18 = v17.rotate(2)
    v19 = v18 * pv0
    v20 = v19.rotate(1)
    v21 = v20.rotate(2)
    v22 = v13 + v21
    v23 = v17.rotate(1)
    v24 = v23 * pv0
    v25 = v24.rotate(2)
    v26 = v22 + v25
    v27 = v23.rotate(2)
    v28 = v27 * pv0
    v29 = v28.rotate(1)
    v30 = v26 + v29
    return [v30]
