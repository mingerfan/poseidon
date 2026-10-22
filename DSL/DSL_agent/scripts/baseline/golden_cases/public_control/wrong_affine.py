def finish(a, weight):
    return a * (weight + 1) + c1

@hc.func("c")
def golden(x):
    weight = c1
    for i in range(3):
        if i == 0:
            break
    else:
        weight = c0
    return finish(x, weight)
