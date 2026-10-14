@hc.func("c")
def golden(x):
    items = [(0, x * c0), (0, x)]
    ordered = sorted(items, key=lambda item: item[0], reverse=True)
    return ordered[0][1] + x + c1
