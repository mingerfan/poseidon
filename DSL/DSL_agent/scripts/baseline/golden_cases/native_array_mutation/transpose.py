@hc.func("c")
def golden(x):
    items = np.array([[x, c0], [x, c1]], dtype=object)
    view = items.T
    view += x
    return items[0, 1] + x * c0 + c1 - c0
