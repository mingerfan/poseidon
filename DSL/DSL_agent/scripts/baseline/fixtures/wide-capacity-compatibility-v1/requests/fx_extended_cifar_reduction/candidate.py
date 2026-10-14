@hc.func('c')
def cap_period_sum_256(a0):
    v1 = a0 + a0.rotate(1)
    v3 = v1 + v1.rotate(2)
    v5 = v3 + v3.rotate(4)
    v7 = v5 + v5.rotate(8)
    v9 = v7 + v7.rotate(16)
    v11 = v9 + v9.rotate(32)
    v13 = v11 + v11.rotate(64)
    return v13 + v13.rotate(128)

@hc.func('c,c,c,c,c,c,c,c,c,c,c,c,c')
def golden(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, zero_ct):
    v0 = x0 + p4
    v1 = x1 + p4
    v2 = x2 + p4
    v3 = x3 + p4
    v4 = x4 + p4
    v5 = x5 + p4
    v6 = x6 + p4
    v7 = x7 + p4
    v8 = x8 + p4
    v9 = x9 + p4
    v10 = x10 + p4
    v11 = x11 + p4
    v55 = cap_period_sum_256(v0) + cap_period_sum_256(v1) + cap_period_sum_256(v2) + cap_period_sum_256(v3) + cap_period_sum_256(v4) + cap_period_sum_256(v5) + cap_period_sum_256(v6) + cap_period_sum_256(v7) + cap_period_sum_256(v8) + cap_period_sum_256(v9) + cap_period_sum_256(v10)
    v56 = cap_period_sum_256(v11)
    return [(cap_period_sum_256(v0) + cap_period_sum_256(v1) + cap_period_sum_256(v2) + cap_period_sum_256(v3) + cap_period_sum_256(v4) + cap_period_sum_256(v5) + cap_period_sum_256(v6) + cap_period_sum_256(v7) + cap_period_sum_256(v8) + cap_period_sum_256(v9) + cap_period_sum_256(v10) + cap_period_sum_256(v11)) * pv0 + ((v55 + v56) * p3).rotate(1).rotate(2).rotate(4).rotate(8).rotate(16).rotate(32).rotate(64).rotate(128) * pv1]
