@hc.func("c")
def pack(value):
    a=np.array([value,value],dtype=object)
    return a*np.array([c0,c0],dtype=object)
@hc.func("c,c")
def combine(a,b):
    return a+b+c1
@hc.func("c")
def golden(x):
    out=pack(x)+np.array([x*.25,x*.25],dtype=object)
    return combine(*out)
