# Manual repair fixture; NEVER counted as Agent generation.
def total(v):
    a=v+v.rotate(1)
    return a+a.rotate(2)
@hc.func("c,c,c,c")
def golden(x,y,z,zero_ct):
    q=float(np.array([p9]).reshape(1))
    w0=np.array([p2,p4,p7,p5]).reshape(4)
    w1=np.array([p8,p3,p5,p5]).reshape(4)
    a0=total(x*w0)+p8
    a1=total(x*w1)+p8
    b0=total(y*w0)*q+p6
    b1=total(y*w1)*q+p6
    out=a1*mask0+a0*mask1+b1*mask2+b0*mask3+total(z)*p10
    linear=a0*mask0+a1*mask1
    return [out,linear]
