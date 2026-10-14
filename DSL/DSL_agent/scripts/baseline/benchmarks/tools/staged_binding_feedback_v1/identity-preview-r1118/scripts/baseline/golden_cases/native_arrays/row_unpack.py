@hc.func("c")
def helper(value):
    return np.array([[value*c0,value],[value+c1,value*2]],dtype=object)
@hc.func("c")
def golden(x):
    top,bottom=helper(x)
    a,b=top
    c,d=bottom
    return a*2+b*.5+c-d*.5
