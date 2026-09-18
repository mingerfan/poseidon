@hc.func("c")
def scale(value):
    return value*c0
@hc.func("c")
def golden(x):
    return scale(x)+scale(x*2)+c1
