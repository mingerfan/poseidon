@hc.func("c")
def golden(x):
    a=np.array([x,x*2],dtype=object)
    view=a[:1]
    b=-view
    a[0]=x*2
    return [b[0]*0.5+view[0]+c1]
