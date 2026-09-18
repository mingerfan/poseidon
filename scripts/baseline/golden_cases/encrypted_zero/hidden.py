@hc.func("c,c")
def golden(x, zero_ct):
    row = x * c0
    pair = row + row.rotate(1)
    total = pair + pair.rotate(2)
    zero_square = zero_ct * zero_ct
    normal_square = total * total
    out = zero_square * c1 + normal_square * c2
    return out + c3
