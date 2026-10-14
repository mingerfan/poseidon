def dot4(v, w):
    p = v * np.array(w)
    s = p + p.rotate(1)
    return s + s.rotate(2)

@hc.func("c")
def golden(x):
    a = dot4(x,[1.,0.,0.,0.])
    b = dot4(x,[0.,0.,1.,0.])
    return [a,b,a*2.+.25,b*2.+.25]
