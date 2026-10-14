@hc.func("c,c")
def golden(x, zero_ct):
    s0 = x + x.rotate(1)
    s1 = s0 + s0.rotate(2)
    s2 = s1 + s1.rotate(4)
    s3 = s2 + s2.rotate(8)
    s4 = s3 + s3.rotate(16)
    s5 = s4 + s4.rotate(32)
    s6 = s5 + s5.rotate(64)
    s7 = s6 + s6.rotate(128)
    return [s7 * p3 - p4]
