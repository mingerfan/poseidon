@hc.func("c")
def golden(x):
    p0 = x * c0
    r0 = p0 + p0.rotate(2)
    h0 = r0 + r0.rotate(1) + c1
    p1 = x * c2
    r1 = p1 + p1.rotate(2)
    h1 = r1 + r1.rotate(1) + c3
    p2 = x * c4
    r2 = p2 + p2.rotate(2)
    h2 = r2 + r2.rotate(1) + c5
    p3 = x * c6
    r3 = p3 + p3.rotate(2)
    h3 = r3 + r3.rotate(1) + c7
    p4 = x * c8
    r4 = p4 + p4.rotate(2)
    h4 = r4 + r4.rotate(1) + c9
    p5 = x * c10
    r5 = p5 + p5.rotate(2)
    h5 = r5 + r5.rotate(1) + c11
    p6 = x * c12
    r6 = p6 + p6.rotate(2)
    h6 = r6 + r6.rotate(1) + c13
    p7 = x * c14
    r7 = p7 + p7.rotate(2)
    h7 = r7 + r7.rotate(1) + c15
    a0 = h0 * h0 + c16
    a1 = h1 * h1 + c17
    a2 = h2 * h2 + c18
    a3 = h3 * h3 + c19
    a4 = h4 * h4 + c20
    a5 = h5 * h5 + c21
    a6 = h6 * h6 + c22
    a7 = h7 * h7 + c23
    y0 = a0 * c24 + a1 * c25 + a2 * c26 + a3 * c27 + a4 * c28 + a5 * c29 + a6 * c30 + a6 * c31 + c32
    y1 = a0 * c33 + a1 * c34 + a2 * c35 + a3 * c36 + a4 * c37 + a5 * c38 + a6 * c39 + a7 * c40 + c41
    return [y0, y1]
