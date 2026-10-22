@hc.func("c")
def golden(x):
    a=np.array([x],dtype=object)
    alias=a
    b=a+0.375
    a*=0.5
    return b[0]+alias[0]

