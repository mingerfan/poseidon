@hc.func("c,c")
def golden(x, zero_ct):
    row = x * c1
    pairs = row + row.rotate(1)
    total = pairs + pairs.rotate(2)
    masked = total * c2
    return [zero_ct, masked]
