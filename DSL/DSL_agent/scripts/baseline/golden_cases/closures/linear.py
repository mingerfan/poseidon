def make_row(weight):
    def dot(value):
        result = value * weight
        for step in [1, 2]:
            result = result + result.rotate(step)
        return result
    return dot

@hc.func("c")
def golden(x):
    first = make_row(c0)
    second = make_row(c1)
    return [first(x), second(x)]
