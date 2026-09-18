@hc.func("c")
def golden(x):
    table = {key: value for key, value in [("scaled", x), ("scaled", x * c0)]}
    keys = list(table)
    return table[keys[0]] + x + c1
