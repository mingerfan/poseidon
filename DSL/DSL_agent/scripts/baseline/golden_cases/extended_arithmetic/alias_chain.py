@hc.func("c")
def golden(x):
    alias = x
    x *= c0
    x += alias
    x = -x
    x -= c1
    return -x
