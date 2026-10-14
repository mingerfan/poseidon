@hc.func('c,c')
def golden(x0, zero_ct):
    v0 = x0.rotate(1)
    v1 = v0.rotate(2)
    v2 = v1 * pv0
    v3 = x0 * pv0
    v4 = v3.rotate(1)
    v5 = v4.rotate(2)
    v6 = v5.rotate(4)
    v7 = v2 + v6
    v8 = v2.rotate(2)
    v9 = v8.rotate(4)
    v10 = v7 + v9
    v11 = v0.rotate(4)
    v12 = v11 * pv0
    v13 = v12.rotate(1)
    v14 = v13.rotate(4)
    v15 = v10 + v14
    v16 = x0.rotate(2)
    v17 = v16.rotate(4)
    v18 = v17 * pv0
    v19 = v18.rotate(4)
    v20 = v15 + v19
    v21 = x0.rotate(4)
    v22 = v21 * pv0
    v23 = v22.rotate(1)
    v24 = v23.rotate(2)
    v25 = v20 + v24
    return [v25]
