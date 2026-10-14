@hc.func("c")
def golden(x):
    acc = x
    for i in range(1, 4):
        acc += x.rotate(i)
    return acc
