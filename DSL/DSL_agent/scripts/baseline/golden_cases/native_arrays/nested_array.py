@hc.func("c")
def first(value):
    return np.array([[value*c0,value],[value+c1,value*2]],dtype=object)
@hc.func("c")
def second(value):
    return first(value)[:,::-1]
@hc.func("c")
def golden(x):
    r=second(x)
    a=r[0,1]
    b=r[0,0]
    c=r[1,1]
    d=r[1,0]
    return a*2+b*.5+c-d*.5
