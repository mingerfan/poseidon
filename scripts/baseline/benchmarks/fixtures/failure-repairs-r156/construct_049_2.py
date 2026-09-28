# Manual repair fixture; NEVER counted as Agent generation.
@hc.func("c,c,c,c,c")
def golden(x, y, z, t, zero_ct):
    T2 = y * y * p10 - p9
    T3 = y * T2 * p10 - y
    coefs = {'a': p6, 'b': p2, 'c': p5, 'd': p4}
    it = coefs.items()
    n1 = zero_ct
    for key, val in it:
        if key == 'a':
            n1 = n1 + val
        elif key == 'b':
            n1 = n1 + val * y
        elif key == 'c':
            n1 = n1 + val * T2
        else:
            n1 = n1 + val * T3
    n0 = x * p7
    n1 = n1 * (mask0 + mask1)
    n2 = n0 + n1.rotate(2)
    A_z = z + z.rotate(2)
    B_z = A_z + A_z.rotate(1)
    mean_z = B_z * p8
    A_t = t + t.rotate(2)
    B_t = A_t + A_t.rotate(1)
    mean_t = B_t * p8
    n4 = n2 + mean_z
    n6 = n4 + mean_t
    return [n6]
