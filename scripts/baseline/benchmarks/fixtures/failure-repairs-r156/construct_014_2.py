# Manual repair fixture; NEVER counted as Agent generation.
@hc.func("c,c,c,c,c")
def golden(x, y, z, t, zero_ct):
    def sum_all(c):
        s1 = c + c.rotate(1)
        s2 = s1 + s1.rotate(2)
        return s2

    cheb = np.polynomial.Chebyshev([0.0, 0.0625], window=[0.5, 2.0])
    half = cheb.window[0]

    y0 = y * mask0 * -0.125
    y1 = y * mask1 * -0.03125
    n1 = sum_all(y0 + y1) + 0.0625

    n2_0 = n1 * -0.25 + 0.03125
    n2_1 = n1 * 0.1875 + 0.03125

    n2_packed = n2_0 * mask2 + n2_1 * mask3

    x_sq = x * x

    M = (sum_all(z) + sum_all(t)) * half

    out = x_sq + n2_packed + M
    return [out]
