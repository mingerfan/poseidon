@hc.func("c")
def golden(x):
    functions = [lambda a: a * weight for weight in [c0, c1]]
    return functions[0](x) + x + c1
