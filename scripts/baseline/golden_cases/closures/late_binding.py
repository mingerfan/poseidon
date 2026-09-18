@hc.func("c")
def golden(x):
    value = x
    def read():
        return value
    saved = read
    value = x * c0
    return saved() + x + c1
