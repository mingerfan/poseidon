@hc.func("c")
def golden(x):
    items = np.array([x], dtype=object)
    alias = items
    for i in range(2):
        items *= c0
    return alias[0] * 6 + c1
