@hc.func("c")
def golden(x):
    gain=np.array([c0[0],c0[0],c0[1],c0[1]])
    bias=np.array([c1[0],c1[0],c1[1],c1[1]])
    return x*gain+bias
