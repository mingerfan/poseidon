# Manual repair fixture; NEVER counted as Agent generation.
@hc.func("c,c,c")
def golden(x, y, zero_ct):
    counter = 0
    c = p6
    for i in range(4):
        if i == 0:
            counter = counter + 1
            break
        c = p3
    c = c * counter
    t = x * p4 + p5
    t = t * x + p2
    t = t * x + c
    yy = y * y
    yy = yy.rotate(4)
    yy = yy.rotate(1)
    t = t * (mask0 + mask1 + mask2)
    out = t + yy
    return (out,)
