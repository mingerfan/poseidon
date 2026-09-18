@hc.func("c")
def helper(value):
    return np.array([[value*c0,value],[value+c1,value*2]],dtype=object).reshape(1,2,1,2).transpose(3,1,0,2)
@hc.func("c")
def golden(x):
    r=helper(x).copy().reshape(4)
    return r[0]*2+r[2]*.5+r[1]-r[3]*.5
