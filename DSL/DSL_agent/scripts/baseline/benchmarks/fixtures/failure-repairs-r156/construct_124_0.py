# Manual repair fixture; NEVER counted as Agent generation.
@hc.func("c,c")
def golden(x, zero_ct):
    cheb = np.polynomial.Chebyshev(coef=[0.078125, -0.11328125, 0.015625, 0.00390625], domain=[-1.0, 1.0], window=[-1.0, 1.0])
    cvec = cheb.coef
    one = zero_ct + 1.0
    T0 = one
    T1 = x
    T2 = 2.0 * x * T1 - T0
    T3 = 2.0 * x * T2 - T1
    result = cvec[0] * T0 + cvec[1] * T1 + cvec[2] * T2 + cvec[3] * T3
    result = result * np.array([1.0]*17+[0.0]*15)
    s1 = result + result.rotate(16)
    s2 = s1 + s1.rotate(8)
    s3 = s2 + s2.rotate(4)
    s4 = s3 + s3.rotate(2)
    s5 = s4 + s4.rotate(1)
    out = s5 * p6
    return [out]