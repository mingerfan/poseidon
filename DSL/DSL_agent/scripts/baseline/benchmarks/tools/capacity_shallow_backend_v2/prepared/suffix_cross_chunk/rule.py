@hc.func('c,c,c')
def golden(x0, x1, zero_ct):
    v0 = x0 * x0
    v1 = x1 * x1
    v2 = v0.rotate(2)
    v3 = v2.rotate(1)
    v4 = v3.rotate(1)
    v5 = v4.rotate(1)
    v6 = v5.rotate(1)
    v7 = v6.rotate(1)
    v8 = v7.rotate(1)
    v9 = v8.rotate(1)
    v11 = v1.rotate(1)
    v12 = v11.rotate(2).rotate(8)
    v13 = v12.rotate(1)
    v14 = v13.rotate(1)
    v15 = v14.rotate(1)
    return [(v0 + v0.rotate(1)) * pv0 + v2 * pv1 + v3 * pv2 + v4 * pv3 + v5 * pv4 + v6 * pv5 + v7 * pv6 + v8 * pv7 + v9 * pv8 + v9.rotate(1) * pv9 + v1 * pv10 + (v11 + v11.rotate(1)) * pv11 + v12 * pv12 + v13 * pv13 + v14 * pv14 + v15 * pv15 + v15.rotate(1) * pv16]
