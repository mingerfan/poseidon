@hc.func("c,c")
def golden(x,zero_ct):
    square = x*x
    total = x
    total = total+total.rotate(1)
    total = total+total.rotate(2)
    total = total+total.rotate(4)
    total = total+total.rotate(8)
    total = total+total.rotate(16)
    total = total+total.rotate(32)
    total = total+total.rotate(64)
    total = total+total.rotate(128)
    return [square,total*p3]
