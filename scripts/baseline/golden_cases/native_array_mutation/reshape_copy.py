@hc.func("c")
def golden(x):
    items = np.array([[x, x], [x, x]], dtype=object)
    view = items.T
    copied = view.reshape(4)
    copied *= c0
    return items[0, 0] + copied[0] + c1
