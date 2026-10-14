@hc.func("c")
def golden(x):
    value = x
    snapshot = value
    def read():
        return snapshot
    saved = read
    value = x * c0
    return saved() + x + c1
