@hc.func("c,c,c,c,c")
def golden(x, y, z, t, zero_ct):
    combined = x + y + z + t
    row = combined * c0
    pair = row + row.rotate(1)
    total = pair + pair.rotate(2)
    return [zero_ct, total]
