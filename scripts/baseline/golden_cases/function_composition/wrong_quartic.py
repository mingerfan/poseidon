def repeated_square(value, depth):
    if depth == 0:
        return value
    return repeated_square(value * value, depth - 1)

@hc.func("c")
def golden(x):
    return repeated_square(x, 1) * c0 + c1
