@hc.func("c")
def golden(x):
    items = np.array([x], dtype=object)
    copied = items.copy()
    flattened = items.flatten()
    items *= c0
    return items[0] + copied[0] * c0 + flattened[0] * c0 + c1
