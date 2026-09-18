@hc.func("c")
def golden(x):
    return affine(x)+x
@hc.func("c")
def affine(value):
    return value*c0+c1
