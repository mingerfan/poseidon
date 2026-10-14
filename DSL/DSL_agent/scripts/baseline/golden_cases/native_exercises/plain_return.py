@hc.func("p")
def identity(value):
    return value
@hc.func("c")
def golden(x):
    return x*identity(c0)+x+c1
