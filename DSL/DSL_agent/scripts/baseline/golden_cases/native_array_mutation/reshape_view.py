@hc.func("c")
def golden(x):
    items = np.array([x, x], dtype=object)
    view = items.reshape(1, 2)
    view *= c0
    return items[0] + items[1] + x * c0 + c1
