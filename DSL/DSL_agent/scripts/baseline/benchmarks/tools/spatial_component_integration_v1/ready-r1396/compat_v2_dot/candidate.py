@hc.func('c,c,c')
def golden(x0, x1, zero_ct):
    v0 = x0 * x1
    v1 = v0 * pv0
    v2 = v0.rotate(1)
    v3 = v2 * pv0
    v4 = v1 + v3
    v5 = v0 + v2
    v6 = v5.rotate(2)
    v7 = v6 * pv0
    v8 = v4 + v7
    return [v8]
