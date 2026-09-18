@hc.func("c,p,c,p")
def combine(a,weight,b,bias):
    return a*weight+b+bias
@hc.func("c")
def golden(x):
    return combine(*(x,),*np.array([c0],dtype=object),x,*(c1,))
