@hc.func("c")
def helper(value):
    return np.array(value*c0+value+c1,dtype=object)
@hc.func("c")
def golden(x):
    return helper(x)
