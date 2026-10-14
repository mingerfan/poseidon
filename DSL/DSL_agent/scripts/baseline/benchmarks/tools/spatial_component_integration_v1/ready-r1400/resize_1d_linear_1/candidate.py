@hc.func('c,c,c,c')
def golden(x0, x1, x2, zero_ct):
    return [x0 * pv0 + x0.rotate(2) * pv1 + x0.rotate(1).rotate(2) * pv2, x0 * pv3 + x0.rotate(1) * pv4 + x0.rotate(2) * pv5 + x1.rotate(1) * pv6 + x1.rotate(2) * pv7, x1 * pv8 + x1.rotate(1) * pv9 + x1.rotate(1).rotate(2) * pv10, x1.rotate(2) * pv11 + x1.rotate(1).rotate(2) * pv12 + x2.rotate(2) * pv13 + x2.rotate(1).rotate(2) * pv14, x2 * pv15 + x2.rotate(1) * pv16 + x2.rotate(2) * pv17]
