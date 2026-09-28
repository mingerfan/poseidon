# Manual repair fixture; NEVER counted as Agent generation.
@hc.func("c,c,c,c")
def golden(x, y, z, zero_ct):
    def lift(v):
        return v.rotate(1).rotate(4)
    scaled = x * p3
    squared = y * y
    second = lift(squared)
    stacked = scaled + second
    t = z + z.rotate(1)
    t = t + t.rotate(2)
    t = t + t.rotate(4)
    mean_all = t * p4
    out0 = stacked + mean_all
    return [out0, scaled]
