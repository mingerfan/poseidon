@hc.func("c")
def golden(x):
    a=np.array([[x],[x*2]],dtype=object)
    b=a*np.asarray([0.5,0.25])
    return b[0,0]+b[1,1]+x*0.5+0.375

