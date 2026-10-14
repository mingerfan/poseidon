@hc.func("c,c")
def golden(x, y):
    difference = x - y
    product = difference * weight
    pair = product + product.rotate(1)
    total = pair + pair.rotate(2)
    return total + bias
