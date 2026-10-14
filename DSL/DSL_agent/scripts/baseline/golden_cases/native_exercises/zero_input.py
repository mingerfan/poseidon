@hc.func("")
def bias():
    return c1
@hc.func("c")
def golden(x):
    return x*c0+x+bias()
