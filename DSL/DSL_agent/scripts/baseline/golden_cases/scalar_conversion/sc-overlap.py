@hc.func("c")
def golden(x):
    invw=float(c0/c0)/float(c0)
    halfb=float(c1)*float(c0)
    a=np.empty(4,dtype=object)
    a[0]=x
    a[1]=c0*x
    a[2]=invw
    a[3]=halfb
    left=a[0:3]
    right=a[1:4]
    left*=right
    return [a[0]+a[1]+a[2]]
