@hc.func("c")
def golden(x):
    a=np.array([x],dtype=object)
    weights=np.array([c0],dtype=object)
    out=a*weights+a+c1
    return out[0]
