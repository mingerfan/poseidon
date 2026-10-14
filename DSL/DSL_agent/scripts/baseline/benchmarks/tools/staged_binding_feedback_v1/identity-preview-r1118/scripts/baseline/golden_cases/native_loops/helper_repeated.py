@hc.func("c")
def half(value):
    return value*c0
@hc.func("c")
def golden(x):
    acc=x
    for i in range(2):
        acc=half(acc)
    return acc*6+c1
