def dot4(v, w):
    p = v * np.array(w)
    s = p + p.rotate(1)
    return s + s.rotate(2)

@hc.func("c,c")
def golden(x,y):
    return [dot4(x,[.5,-1.,.25,2.])+dot4(y,[-2.,.75,1.,-.5]),dot4(x,[1.,2.,3.,4.])+dot4(y,[4.,3.,2.,1.])]
