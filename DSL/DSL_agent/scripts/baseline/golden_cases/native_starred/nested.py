@hc.func("c,p,c,p")
def combine(a,weight,b,bias):
    return a*weight+b+bias
@hc.func("c")
def pack(value):
    return np.array([value,c0,value,c1],dtype=object)
@hc.func("c")
def golden(x):
    return combine(*pack(x))
