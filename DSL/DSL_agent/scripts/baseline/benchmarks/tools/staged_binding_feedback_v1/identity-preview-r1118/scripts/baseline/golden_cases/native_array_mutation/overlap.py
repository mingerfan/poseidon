@hc.func("c")
def golden(x):
    items = np.array([x, x * 2, x * 3], dtype=object)
    left = items[1:]
    right = items[:-1]
    left -= right
    return items[1] + items[2] * c0 + c1
