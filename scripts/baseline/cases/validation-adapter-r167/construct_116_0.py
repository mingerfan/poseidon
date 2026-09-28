def rot(c, k):
    if k == 0:
        return c
    if k == 1:
        return c.rotate(1)
    if k == 2:
        return c.rotate(2)
    if k == 3:
        return c.rotate(1).rotate(2)
    if k == 4:
        return c.rotate(4)
    if k == 5:
        return c.rotate(1).rotate(4)
    if k == 6:
        return c.rotate(2).rotate(4)
    if k == 7:
        return c.rotate(1).rotate(2).rotate(4)
    return c

@hc.func("c,c,c,c")
def golden(x, y, z, zero_ct):
    coef_map = {'k0': p6, 'k1': p2, 'k2': p5, 'k3': p4}
    coef_view = coef_map.values()
    coefs = []
    for item in coef_view:
        coefs.append(item)

    k0 = coefs[0]
    k1 = coefs[1]
    k2 = coefs[2]
    k3 = coefs[3]

    n0 = k0 + k1 * x + k2 * x * x + k3 * x * x * x

    n1 = y + p7

    m0 = np.array(mask0)
    m1 = np.array(mask1)
    m2 = np.array(mask2)
    m3 = np.array(mask3)
    m4 = np.array(mask4)
    m5 = np.array(mask5)
    m012 = m0 + m1 + m2
    m345 = m3 + m4 + m5
    m06 = m012 + m345

    n2 = n0 * m012 + rot(n1, 5) * m345

    z_sum = z + rot(z, 1) + rot(z, 2)
    mean = z_sum * p8 * m0
    mean_b = mean
    mean_b = mean_b + rot(mean_b, 1)
    mean_b = mean_b + rot(mean_b, 2)
    mean_b = mean_b + rot(mean_b, 4)

    n4 = (n2 + mean_b) * m06

    output0 = rot(n4, 1) * (m0 + m1 + m2 + m3 + m4) + rot(n4 * m0, 3) * m5
    output1 = n0 * m012

    return [output0, output1]
