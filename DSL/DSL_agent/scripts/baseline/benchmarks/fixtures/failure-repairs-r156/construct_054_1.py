@hc.func("c,c,c,c")
def golden(x, y, z, zero_ct):
    x2 = x * x
    x3 = x2 * x
    poly = x3 * (4.0 * p6) + x2 * (2.0 * p7) + x * (p3 - 3.0 * p6) + (p8 - p7)
    n0v0 = poly * mask0 + poly * mask1
    t = y * p3 * mask0 + y * p4 * mask1
    s = t + t.rotate(1) + t.rotate(2) + t.rotate(1).rotate(2)
    n1 = s + p8
    A = n1 * p2 + p7
    B = n1 * p9 + p7
    n2v0 = A * mask2 + B * mask3
    n3v0 = n0v0 + n2v0
    zs = z + z.rotate(1) + z.rotate(2) + z.rotate(1).rotate(2)
    m = zs * p10
    out = n3v0 + m
    return list(iter([out]))