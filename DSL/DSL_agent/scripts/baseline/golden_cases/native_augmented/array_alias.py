@hc.func("c")
def golden(x):
    items = np.array([x], dtype=object)
    value = items[0]
    value *= c0
    value += items[0]
    value += c1
    return value
