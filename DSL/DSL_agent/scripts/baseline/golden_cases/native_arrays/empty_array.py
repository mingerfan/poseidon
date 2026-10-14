@hc.func("c")
def helper(value):
    return np.array([],dtype=object).reshape(0,2)
@hc.func("c")
def golden(x):
    helper(x)
    return x*c0+x+c1
