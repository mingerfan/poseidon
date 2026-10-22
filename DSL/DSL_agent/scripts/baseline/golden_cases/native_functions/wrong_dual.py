@hc.func("c,c")
def difference(first,second):
    return second-first

@hc.func("c,c")
def golden(x,y):
    return difference(x,y)
