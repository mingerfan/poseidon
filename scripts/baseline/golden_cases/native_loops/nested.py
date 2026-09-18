@hc.func("c")
def golden(x):
    acc=x*.75
    for i in range(2):
        for j in range(i+1):
            acc=acc+x*.25
    return acc+c1
