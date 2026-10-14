def finish(a):
    return a + c1

@hc.func("c")
def golden(x):
    groups = [[x * c0]] * 2
    groups[1][:] = [x * c0 + x]
    return finish(groups[0][0])
