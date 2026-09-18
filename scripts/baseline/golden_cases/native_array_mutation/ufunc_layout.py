@hc.func("c")
def golden(x):
    items = np.array([[x, x], [x, x]], dtype=object)
    transposed = items.T
    product = transposed * c0
    flattened = product.reshape(4)
    flattened *= c0
    return product[0, 0] + flattened[0] * 4 + c1
