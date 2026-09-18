@hc.func("c")
def golden(x):
    result=[]
    for i in range(4):
        mask=[0.0]*4
        mask[i]=1.0
        v=x*np.array(mask)
        v=v+v.rotate(1)
        v=v+v.rotate(2)
        result.append(v*(-0.5)+0.375)
    return result
