@hc.func("c,p")
def helper(value,weight):
    return np.array([[value,weight],[value*c0,value+c1]],dtype=object)
@hc.func("c")
def golden(x):
    r=helper(x,c0)
    return r[0,0]*r[0,1]+r[1,1]
