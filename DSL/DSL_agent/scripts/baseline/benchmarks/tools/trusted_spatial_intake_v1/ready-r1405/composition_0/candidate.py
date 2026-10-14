@hc.func('c,c')
def golden(x0, zero_ct):
    v6 = x0 * pv0 + x0.rotate(2) * pv1 + x0.rotate(1).rotate(2) * pv2
    v8 = x0.rotate(2) * pv3
    return [v6 * pv4 + v6.rotate(1) * pv5 + v6.rotate(2) * pv6 + v6.rotate(1).rotate(2) * pv7 + v8 * pv8 + v8.rotate(1).rotate(2) * pv9 + pv10]
