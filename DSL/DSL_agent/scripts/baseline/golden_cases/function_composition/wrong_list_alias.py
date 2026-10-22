def scale_first(items, weight):
    items = [items[0] * weight]
    return

def finish(items, original, bias):
    for index in range(3):
        if index == 1:
            return items[0] + original + bias
    return original

@hc.func("c")
def golden(x):
    items = [x]
    scale_first(items, c0)
    return finish(items, x, c1)
