def make_row(weight, steps=(1, 2)):
    def dot(value, row=weight, *, rotations=steps):
        result = value * row
        for step in rotations:
            result = result + result.rotate(step)
        return result
    return dot

@hc.func("c")
def golden(x):
    first = make_row(c0)
    second = make_row(weight=c1)
    return [first(x), second(value=x)]
