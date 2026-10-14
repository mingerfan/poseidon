@hc.func("c")
def golden(x):
    values = [x]
    pairs = zip(values, [x])
    values[0] = x * c0
    results = [a + b + c1 for a, b in pairs]
    return results[0]
