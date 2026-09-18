@hc.func("c")
def golden(x):
    weights = (c0, c1)
    outputs = []
    for row in range(len(weights)):
        total = x * weights[row]
        for step in range(1, 2):
            total += total.rotate(step)
        outputs.append(total)
    return outputs
