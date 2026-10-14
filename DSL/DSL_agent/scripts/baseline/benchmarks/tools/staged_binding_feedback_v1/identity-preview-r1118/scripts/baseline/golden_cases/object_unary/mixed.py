@hc.func("c")
def golden(x):
    a=np.array([x,0.5,True],dtype=object)
    b=-a
    return [(b[0]*0.5+x*2)*(0-b[2])+b[1]+0.875]
