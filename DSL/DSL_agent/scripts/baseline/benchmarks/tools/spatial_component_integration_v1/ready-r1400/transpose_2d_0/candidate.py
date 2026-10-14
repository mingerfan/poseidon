@hc.func('c,c,c')
def golden(x0, x1, zero_ct):
    return [x0 * pv0 + x0.rotate(1).rotate(2) * pv1, x0 * pv2 + x0.rotate(2) * pv3 + x0.rotate(1).rotate(2) * pv4 + x1.rotate(2) * pv14 + x1.rotate(1).rotate(2) * pv15, x0.rotate(2) * pv5 + x0.rotate(1).rotate(2) * pv6 + x1.rotate(2) * pv16 + x1.rotate(1).rotate(2) * pv17, x0 * pv7 + x0.rotate(1).rotate(2) * pv8, x0 * pv9 + x0.rotate(2) * pv10 + x0.rotate(1).rotate(2) * pv11 + x1.rotate(2) * pv18 + x1.rotate(1).rotate(2) * pv19, x0.rotate(2) * pv12 + x0.rotate(1).rotate(2) * pv13 + x1.rotate(2) * pv20 + x1.rotate(1).rotate(2) * pv21]
