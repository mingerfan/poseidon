def affine(a, weight, bias):
    return a * weight + bias

@hc.func("c")
def golden(x):
    weight = c0[0] + float("0.5")
    bias = float("3") / 8
    return affine(x, weight, bias)
