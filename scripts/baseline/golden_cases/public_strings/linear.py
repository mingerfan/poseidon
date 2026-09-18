def dot(a, weight):
    v = a*weight
    v = v+v.rotate(1)
    return v+v.rotate(2)

def rows(texts):
    return [[float(token) for token in row.strip().split(",")] for row in texts]

@hc.func("c")
def golden(x):
    weights = rows([" 0.5,-0.25,0.125,0.75 "," -0.375,0.25,0.5,-0.125 "])
    return [dot(x,weight) for weight in weights]
