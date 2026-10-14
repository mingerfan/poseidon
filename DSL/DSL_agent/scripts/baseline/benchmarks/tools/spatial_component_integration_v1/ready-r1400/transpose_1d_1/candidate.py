@hc.func('c,c,c,c')
def golden(x0, x1, x2, zero_ct):
    return [x0 * pv0 + x0.rotate(2) * pv1 + x0.rotate(1).rotate(2) * pv2 + x1.rotate(1) * pv4 + x1.rotate(2) * pv5 + pv15, x0.rotate(1) * pv3 + x0.rotate(2) * pv0 + x1 * pv6 + x1.rotate(1) * pv7 + x1.rotate(1).rotate(2) * pv8 + x2.rotate(2) * pv11 + x2.rotate(1).rotate(2) * pv12 + pv16, x1.rotate(2) * pv9 + x1.rotate(1).rotate(2) * pv10 + x2 * pv11 + x2.rotate(1) * pv13 + x2.rotate(2) * pv14 + pv17]
