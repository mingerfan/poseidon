@hc.func("c")
def golden(x):
    return second(x,c0,c1)+x

@hc.func("c,p,p")
def second(value,weight,bias):
    return first(value,weight)+bias

@hc.func("c,p")
def first(value,weight):
    return value*weight
