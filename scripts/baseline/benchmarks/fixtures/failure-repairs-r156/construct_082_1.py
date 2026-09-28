# Manual repair fixture; NEVER counted as Agent generation.
def vec4(a, b, c, d):
    return np.array([a, b, c, d]).reshape(4)


def rot_sum(c):
    r1 = c.rotate(1)
    r2 = c.rotate(2)
    r3 = r1.rotate(2)
    return c + r1 + r2 + r3


def update(current, condition, new_value):
    if condition:
        return new_value
    return current


@hc.func("c,c,c,c,c")
def golden(x, y, z, t, zero_ct):
    w0 = vec4(p2, p4, p7, p5)
    w1 = vec4(p8, p3, p5, p5)
    m0 = vec4(1.0, 0.0, 0.0, 0.0)
    m1 = vec4(0.0, 1.0, 0.0, 0.0)

    n0 = rot_sum(x * w0) * m0 + rot_sum(x * w1) * m1 + p8

    yy = y * y
    n2 = rot_sum(yy * w0) * m0 + rot_sum(yy * w1) * m1 + p6

    n3 = n0 * n2
    n4 = rot_sum(z) * p9
    n6 = rot_sum(t) * p9
    n5 = n3 + n4
    n7 = n5 + n6

    state = {"out": zero_ct}
    state.update({"out": n7})

    return [state["out"]]
