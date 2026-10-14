@hc.func('c,c,c')
def golden(x0, x1, zero_ct):
    return [x0 * pv0 + x0.rotate(2) * pv1 + x0.rotate(1).rotate(2) * pv2 + x1.rotate(1) * pv9 + x1.rotate(2) * pv10 + x1.rotate(1).rotate(2) * pv11 + pv16, x0 * pv3 + x0.rotate(1) * pv4 + x0.rotate(2) * pv5 + x1 * pv12 + x1.rotate(1) * pv13 + pv17, x0 * pv6 + x0.rotate(1) * pv7 + x0.rotate(1).rotate(2) * pv8 + x1 * pv14 + x1.rotate(2) * pv3 + x1.rotate(1).rotate(2) * pv15 + pv18]
