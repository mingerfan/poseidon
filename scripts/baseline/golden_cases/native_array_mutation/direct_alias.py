@hc.func("c")
def golden(x):
    items = np.array([x], dtype=object)
    alias = items
    items *= c0
    return alias[0] + x + c1
