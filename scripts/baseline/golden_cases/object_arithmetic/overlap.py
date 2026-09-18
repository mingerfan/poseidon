@hc.func("c")
def golden(x):
    a=np.array([x,x*0.5,x*0.5],dtype=object)
    a[1:]+=a[:-1]
    return (a[1]+a[2])*0.6+0.375

