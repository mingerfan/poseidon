def dot4(v, w):
    p = v * np.array(w)
    s = p + p.rotate(1)
    return s + s.rotate(2)

@hc.func("c")
def golden(x):
    return [dot4(x,[1.,0.,0.,0.]),dot4(x,[0.,0.,1.,0.]),dot4(x,[0.,1.,0.,0.]),dot4(x,[0.,0.,0.,1.])]
