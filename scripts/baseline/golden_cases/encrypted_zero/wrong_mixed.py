@hc.func("c,c")
def golden(x, zero_ct):
    constant = x + c0
    row = x * c1
    pair = row + row.rotate(1)
    total = pair + pair.rotate(2)
    normal = total + c2
    return [constant, normal]
