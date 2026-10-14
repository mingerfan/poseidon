@hc.func("c,c")
def golden(x, zero_ct):
    u = x * x
    r3 = u.rotate(2)
    r4 = r3.rotate(4)
    r5 = r4.rotate(8)
    r6 = r5.rotate(16)
    r7 = r6.rotate(32)
    r8 = r7.rotate(64)
    r9 = r8.rotate(128)
    r10 = u.rotate(1)
    r11 = r10.rotate(2)
    r12 = r11.rotate(4)
    r13 = r12.rotate(8)
    r14 = r13.rotate(16)
    r15 = r14.rotate(32)
    r16 = r15.rotate(64)
    a = r16 * mask0
    r18 = u.rotate(128)
    b = r18 * mask0
    r20 = b.rotate(1)
    r21 = r20.rotate(2)
    r22 = r21.rotate(4)
    r23 = r22.rotate(8)
    r24 = r23.rotate(16)
    r25 = r24.rotate(32)
    r26 = r25.rotate(64)
    r27 = r26.rotate(128)
    return [r9 + a + r27]
