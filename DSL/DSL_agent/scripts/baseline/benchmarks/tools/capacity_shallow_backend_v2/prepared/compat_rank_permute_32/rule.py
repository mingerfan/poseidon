@hc.func('c,c')
def golden(x0, zero_ct):
    v0 = x0.rotate(1)
    v1 = v0.rotate(1)
    v2 = v1.rotate(1)
    v3 = v2.rotate(1)
    v4 = v3.rotate(1)
    v5 = v4.rotate(1)
    v6 = v5.rotate(1)
    v7 = v6.rotate(2)
    v8 = v7.rotate(1)
    v9 = v8.rotate(1)
    v10 = v9.rotate(1)
    v11 = v10.rotate(1)
    v12 = v11.rotate(1)
    v13 = v12.rotate(1)
    v27 = (x0 * pv0 + (v3 * pv1 + v8 * pv2) + (v11 * pv3 + (v12 * pv4 + v13 * pv5)) + (v0 * pv6 + v1 * pv7 + (v2 * pv8 + v4 * pv9) + (v6 * pv10 + v9 * pv11 + (v11 * pv12 + v13 * pv13))).rotate(16) + (v0 * pv14 + v2 * pv15 + (v4 * pv16 + v7 * pv17) + (v9 * pv18 + v11 * pv19 + (v12 * pv20 + v13 * pv21))).rotate(32) + (v0 * pv22 + v1 * pv23 + (v2 * pv24 + (v5 * pv25 + v10 * pv26))).rotate(16).rotate(32)) * pv27 + x0 * x0 * p3
    return [v27, (v27 + v27.rotate(8) + v27.rotate(32) + v27.rotate(8).rotate(32)) * pv28 + (v27.rotate(2) + v27.rotate(2).rotate(8) + v27.rotate(2).rotate(32) + v27.rotate(2).rotate(8).rotate(32)) * pv29 + (v27.rotate(4) + v27.rotate(4).rotate(32)) * pv30 + (v27.rotate(4).rotate(8) + v27.rotate(4).rotate(8).rotate(32)) * pv31 + (v27.rotate(2).rotate(4).rotate(8) + v27.rotate(2).rotate(4).rotate(16) + v27.rotate(2).rotate(4).rotate(8).rotate(32) + v27.rotate(2).rotate(4).rotate(16).rotate(32)) * pv32 + (v27.rotate(16) + v27.rotate(8).rotate(16) + v27.rotate(16).rotate(32) + v27.rotate(8).rotate(16).rotate(32)) * pv33 + (v27.rotate(4).rotate(16) + v27.rotate(4).rotate(16).rotate(32)) * pv34]
