@hc.func("c,p,c,p")
def combine(a,weight,b,bias):
    return a*weight+b+bias
@hc.func("c")
def golden(x):
    items=[x,c0,x,c1]
    return combine(*items)
