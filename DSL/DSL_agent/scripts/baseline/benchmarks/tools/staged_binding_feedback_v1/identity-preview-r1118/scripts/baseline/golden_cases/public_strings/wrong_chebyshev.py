def coefficients(texts):
    return [float(token.strip(" -")) for token in texts]

def build(a, coef):
    terms = {(0,0):1, (1,0):a}
    for degree in range(2,4):
        terms[(degree,0)] = 2*a*terms[(degree-1,0)] - terms[(degree-2,0)]
    return -coef[3]*terms[(3,0)]

@hc.func("c")
def golden(x):
    coef = coefficients([" 0 "," 0 "," 0 "," -1 "])
    return build(x,coef)
