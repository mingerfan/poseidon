@hc.func("c")
def empty(value):
    return []
@hc.func("c")
def golden(x):
    empty(x)
    return x*c0+x+c1
