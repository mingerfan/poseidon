def dot(a, weight):
    value = a * weight
    value = value + value.rotate(1)
    return value + value.rotate(2)

@hc.func("c")
def golden(x):
    rows = {0: c0, 1: c1}
    outputs = []
    for i in range(3):
        if i not in rows:
            break
        outputs.append(dot(x, rows[i]))
    else:
        return [-x, -x]
    return outputs
