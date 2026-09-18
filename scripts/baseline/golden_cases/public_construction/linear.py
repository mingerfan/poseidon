@hc.func("c")
def golden(x):
    weights = (c0, c1)
    outputs = []
    for row in range(len(weights)):
        weight = weights[0] if row == 0 else weights[1]
        total = x * weight
        for step in range(1, 3):
            total += total.rotate(step)
        outputs.append(total)
    return outputs
