@hc.func("c")
def golden(x):
    row = x * c0
    pair = row + row.rotate(1)
    total = pair + pair.rotate(2)
    negative = -total
    centered = negative + c1
    return -centered
