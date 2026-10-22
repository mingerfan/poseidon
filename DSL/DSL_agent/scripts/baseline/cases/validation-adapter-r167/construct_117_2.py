@hc.func("c,c,c")
def golden(x, y, zero_ct):
    counter = {}
    counter["mapping_writes"] = 0
    d = {}
    for step in range(2):
        counter["mapping_writes"] = counter["mapping_writes"] + 1
        if counter["mapping_writes"] == 1:
            d.update({"n0": p6 + p2 * x + p5 * x * x + p4 * x * x * x})
        else:
            d.update({"n1": p7 * y})
    a = d["n0"]
    b = d["n1"]
    out = a * mask0 + a * mask1 + a * mask2 + b.rotate(1).rotate(4)
    return [out]
