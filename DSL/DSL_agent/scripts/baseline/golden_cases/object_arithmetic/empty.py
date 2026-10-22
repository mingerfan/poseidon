@hc.func("c")
def golden(x):
    a=np.full((2,),Empty(),dtype=object)
    b=np.array([x,x*0.5],dtype=object)
    a-=b
    return a[0]+a[1]+0.375

