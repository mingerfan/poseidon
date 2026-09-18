def parse(text):
    tokens = text.split(" ")
    weight = float(tokens[0]) if tokens[1] == "" else 0.25
    return weight,float(tokens[-1])

@hc.func("c")
def golden(x):
    weight,bias = parse("0.5  0.375")
    return x*(weight+1)+bias
