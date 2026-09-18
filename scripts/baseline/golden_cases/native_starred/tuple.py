@hc.func("c,p,c,p")
def combine(a,weight,b,bias):
    return a*weight+b+bias
@hc.func("c")
def golden(x):
    return combine(*(x,c0,x,c1))
