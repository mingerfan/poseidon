@hc.func("c")
def helper(value):
    acc=value
    for step in range(1,4):
        acc=acc+value.rotate(step)
    return acc
@hc.func("c")
def golden(x):
    return helper(x)
