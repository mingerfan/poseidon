@hc.func("c")
def first(value):
    return value*c0
@hc.func("c")
def second(value):
    return first(value)+c1
@hc.func("c")
def golden(x):
    return second(x)+x
