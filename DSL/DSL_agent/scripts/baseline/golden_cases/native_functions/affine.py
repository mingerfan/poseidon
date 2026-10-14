@hc.func("c,p,p")
def affine(value, weight, bias):
    return value*weight+bias

@hc.func("c")
def golden(x):
    return affine(x,c0,c1)+x
