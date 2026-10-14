@hc.func("c")
def identity(value):
    return value
@hc.func("c")
def golden(x):
    value=identity(x)
    return value*c0+value+c1
