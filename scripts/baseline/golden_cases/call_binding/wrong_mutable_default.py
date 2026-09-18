def factory():
    def first(a, items=[]):
        items = []
        items.append(a)
        return items[0]
    return first

@hc.func("c")
def golden(x):
    first = factory()
    original = first(x)
    shared = first(-x)
    return original * c0 + shared + c1
