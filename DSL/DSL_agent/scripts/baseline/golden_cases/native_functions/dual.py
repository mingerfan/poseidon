@hc.func("c,c")
def difference(first,second):
    return first-second

@hc.func("c,c")
def golden(x,y):
    return difference(x,y)
