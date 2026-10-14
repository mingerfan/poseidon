@hc.func('c,c')
def golden(x0, zero_ct):
    v0 = x0.rotate(4)
    v1 = v0 * pv0
    v2 = x0 * pv0
    v3 = v2.rotate(1)
    v4 = v3.rotate(2)
    v5 = v4.rotate(4)
    v6 = v1 + v5
    v7 = v1.rotate(2)
    v8 = v7.rotate(4)
    v9 = v6 + v8
    v10 = x0.rotate(2)
    v11 = v10 * pv0
    v12 = v11.rotate(1)
    v13 = v12.rotate(4)
    v14 = v9 + v13
    return [v14]
