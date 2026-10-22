@hc.func("c,c")
def golden(x, zero_ct):
    u = x * x
    r3 = u.rotate(2)
    r4 = r3.rotate(4)
    r5 = r4.rotate(8)
    r6 = r5.rotate(16)
    r7 = r6.rotate(32)
    r8 = r7.rotate(64)
    r9 = u.rotate(1)
    r10 = r9.rotate(2)
    r11 = r10.rotate(4)
    r12 = r11.rotate(8)
    r13 = r12.rotate(16)
    r14 = r13.rotate(32)
    a = r14 * mask0
    b = r14 * mask1
    return [r8 + a + b]
