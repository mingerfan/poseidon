@hc.func("c,c")
def golden(x, y):
    difference = x - y
    row0 = difference * c1
    half0 = row0 + row0.rotate(2)
    sum0 = half0 + half0.rotate(1)
    output0 = sum0 + c2
    row1 = difference * c3
    half1 = row1 + row1.rotate(2)
    sum1 = half1 + half1.rotate(1)
    output1 = sum1 + c4
    return [output0 * output0, output1 * output1]
