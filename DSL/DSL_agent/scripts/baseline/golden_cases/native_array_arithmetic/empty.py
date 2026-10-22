@hc.func("")
def empty():
    a=np.array([],dtype=object).reshape(0,2)
    return -a+np.array([c0,c1],dtype=object)
@hc.func("c")
def golden(x):
    empty()
    return x*c0+x+c1
