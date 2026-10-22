def affine(a, weight, bias):
    return a * weight + bias

@hc.func("c")
def golden(x):
    column = np.array([[c0[0]], [c0[0]]])
    weight = (column + np.array([1.0, 1.0])).reshape(4)
    return affine(x, weight, c1[0])
