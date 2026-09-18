def dot4(v, w):
    p = v * np.array(w)
    s = p + p.rotate(1)
    return s + s.rotate(2)

@hc.func("c,c")
def golden(x,y):
    return [dot4(x,[1.,2.,3.,4.])+dot4(y,[5.,6.,7.,8.]),dot4(x,[-1.,.5,-.25,1.])+dot4(y,[2.,-2.,.75,-.5])]
