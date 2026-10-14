def reduce4(value):
    for step in range(1, 3):
        value += value.rotate(step)
    return value

def linear(value, weights, reducer):
    outputs = []
    for weight in weights:
        outputs.append(reducer(value * weight))
    return outputs

@hc.func("c")
def golden(x):
    return linear(x, (c0, c1), reduce4)
