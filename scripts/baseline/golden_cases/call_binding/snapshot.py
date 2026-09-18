@hc.func("c")
def golden(x):
    value = x * c0
    def calculate(a=value, *, bias=c1):
        return a + x + bias
    value = x
    return calculate()
