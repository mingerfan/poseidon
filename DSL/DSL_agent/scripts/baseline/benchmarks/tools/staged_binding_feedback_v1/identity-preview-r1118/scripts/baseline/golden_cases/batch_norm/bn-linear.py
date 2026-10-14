@hc.func("c")
def golden(x):
    normalized=x*c0+c1
    left=normalized*c2
    left=left+left.rotate(1)
    left=left+left.rotate(2)
    right=normalized*c4
    right=right+right.rotate(1)
    right=right+right.rotate(2)
    return [left+c3,right+c5]
