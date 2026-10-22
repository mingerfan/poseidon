@hc.func("c,p,p")
def split(value, weight, bias):
    return value*weight,value+bias

@hc.func("c")
def golden(x):
    first,second=split(x,c0,c1)
    return [first+second]
