@hc.func("c")
def helper(value):
    return np.array([[value*c0,value],[value+c1,value*2]],dtype=object).T
@hc.func("c")
def golden(x):
    r=helper(x).flatten()
    return r[0]*2+r[1]*.5+r[2]-r[3]*.5
