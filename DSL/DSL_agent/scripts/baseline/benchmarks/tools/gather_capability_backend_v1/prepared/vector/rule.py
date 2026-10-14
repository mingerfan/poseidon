@hc.func("c,c")
def golden(x0, zero_ct):
    r7 = x0.rotate(1).rotate(2).rotate(4)
    r4 = x0.rotate(4)
    r2 = x0.rotate(2)
    out = r4 * pv0 + r7 * pv1 + r2 * pv2 + r7 * pv3
    return out
