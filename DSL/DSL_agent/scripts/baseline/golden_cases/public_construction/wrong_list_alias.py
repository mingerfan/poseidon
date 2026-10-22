@hc.func("c")
def golden(x):
    values = [x]
    shared = [x]
    for weight in (c0,):
        values[0] = values[0] * weight
    result = shared[-1] + x
    for i in range(2):
        if i == 1:
            result += c1
    return result
