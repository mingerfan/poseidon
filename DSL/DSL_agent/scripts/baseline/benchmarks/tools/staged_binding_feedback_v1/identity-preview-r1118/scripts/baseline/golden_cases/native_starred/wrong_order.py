@hc.func("c,c,p")
def combine(a,b,bias):
    return a-b+bias
@hc.func("c")
def golden(x):
    items=np.array([x*.5,x*2],dtype=object)
    return combine(*items,c1)
