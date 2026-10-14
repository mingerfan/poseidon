@hc.func("c")
def golden(x):
    a=x*np.array([1.0,0.0,0.0,0.0])
    a=a+a.rotate(1)
    a=a+a.rotate(2)
    b=x*np.array([0.0,0.0,1.0,0.0])
    b=b+b.rotate(1)
    b=b+b.rotate(2)
    return [a*(-1.0)+0.375,b*4.0+0.625,a*(-1.0)+0.375,b*4.0+0.625]
