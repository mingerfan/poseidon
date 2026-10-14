@hc.func("c")
def golden(x):
    gain=np.array([-0.8960369509591836,1.8294146962480904]*2)
    bias=np.array([0.125-gain[0]*0.5,0.375+gain[1]*0.25]*2)
    return x*gain+bias
