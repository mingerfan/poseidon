@hc.func("c,p,c,p")
def combine(a,weight,b,bias):
    return a*weight+b+bias
@hc.func("c")
def golden(x):
    items=np.array([x,c0,x,c1],dtype=object)
    return combine(*items)
