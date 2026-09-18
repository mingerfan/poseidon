@hc.func("c")
def golden(x):
    centered = x + c0
    negative = centered * c1
    row0 = negative * c2
    pair0 = row0 + row0.rotate(1)
    total0 = pair0 + pair0.rotate(2)
    out0 = total0 + c3
    row1 = negative * c4
    pair1 = row1 + row1.rotate(1)
    total1 = pair1 + pair1.rotate(2)
    out1 = total1 + c5
    return [out0, out1]
