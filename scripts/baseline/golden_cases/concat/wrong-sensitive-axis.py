def dot4(v, w):
    p = v * np.array(w)
    s = p + p.rotate(1)
    return s + s.rotate(2)

@hc.func("c,c")
def golden(x,y):
    return [dot4(x,[1.,2.,5.,6.])+dot4(y,[3.,4.,7.,8.]),dot4(x,[-1.,.5,2.,-2.])+dot4(y,[-.25,1.,.75,-.5])]
