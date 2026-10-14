def build(a):
    terms = {(0,0): 1, (1,0): a}
    degree = 2
    while degree <= 3:
        previous = (degree-1,0)
        before = (degree-2,0)
        if previous not in terms or before not in terms:
            return -a
        terms[(degree,0)] = 2*a*terms[previous] - terms[before]
        degree += 1
    else:
        return terms[(3,0)]

@hc.func("c")
def golden(x):
    return build(x)
