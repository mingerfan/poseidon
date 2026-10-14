# Manual repair fixture; NEVER counted as Agent generation.
@hc.func("c,c")
def golden(x, zero_ct):
    a = x * mask0
    b = x.rotate(1) * mask0
    inner = a * p2 + b * p3 + zero_ct + mask0 * p5
    em = np.empty(1, dtype=object)
    em[0] = hc.Empty()
    ct = np.empty(1, dtype=object)
    ct[0] = -inner
    res = em - ct
    return res
