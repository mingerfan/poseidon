@hc.func("c")
def golden(x):
    items = [(0, x * c0), (0, x)]
    ordered = list(reversed(sorted(items, key=lambda item: item[0])))
    return ordered[0][1] + x + c1
