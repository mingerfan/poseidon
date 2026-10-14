@hc.func("c")
def split(value):
    return value*c0,value+c1
@hc.func("c")
def golden(x):
    a,b=split(x)
    return a+b
