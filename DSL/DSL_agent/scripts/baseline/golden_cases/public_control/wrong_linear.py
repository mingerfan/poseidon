def dot(a, weight):
    value = a * weight
    value = value + value.rotate(1)
    return value + value.rotate(2)

@hc.func("c")
def golden(x):
    rows = {0: c1, 1: c0}
    outputs = []
    for i in range(3):
        if i not in rows:
            break
        outputs.append(dot(x, rows[i]))
    else:
        return [-x, -x]
    return outputs
