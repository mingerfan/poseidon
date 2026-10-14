@hc.func("c,c")
def golden(x, zero_ct):
    values = np.array([x*x], dtype=object)
    negative = -values
    return -negative[0]
