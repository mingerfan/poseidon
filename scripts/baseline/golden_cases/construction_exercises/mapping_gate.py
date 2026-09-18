@hc.func("c")
def golden(x):
    d = dict(weight=c0, bias=c1)
    w = d.setdefault("selected", d.get("weight"))
    return x * w + x + d.get("bias")
