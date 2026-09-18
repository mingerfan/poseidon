def affine(a, /, *terms, bias, **options):
    result = a * options["weight"]
    for value in terms:
        result = result + value
    return result - bias

def forward(*args, **kwargs):
    return affine(*args, **kwargs)

@hc.func("c")
def golden(x):
    return forward(x, x, bias=c1, weight=c0)
