@hc.func('c,c')
def golden(x0, zero_ct):
    v0 = x0.rotate(4)
    v1 = v0 * pv0
    v2 = x0 * pv0
    v3 = v2.rotate(1)
    v4 = v3.rotate(2)
    v5 = v4.rotate(4)
    v6 = v5.rotate(8)
    v7 = v1 + v6
    v8 = x0.rotate(8)
    v9 = v8 * pv0
    v10 = v9.rotate(2)
    v11 = v10.rotate(4)
    v12 = v11.rotate(8)
    v13 = v7 + v12
    v14 = x0.rotate(2)
    v15 = v14.rotate(8)
    v16 = v15 * pv0
    v17 = v16.rotate(1)
    v18 = v17.rotate(4)
    v19 = v18.rotate(8)
    v20 = v13 + v19
    return [v20]
