"""Equivalent manual expressions for the frozen Attention decode failure."""
from benchmark_graph import digest,require
from rmsnorm_expression_probe import main

MODEL_SHA256="89ca5f3dd294ff3bc065e453c76b19231501b92687b7d8a669fd5271c0710138"

def variants(request):
    require(digest(request["model"])==MODEL_SHA256,"Original Attention model")
    require(request["layout"]["input_slot_period"]==4,"Original Attention packing")
    c=request["public_constants"]
    def scalar(value):
        return next(n for n,v in c.items() if type(v) in (int,float) and v==value)
    projection=scalar(.2);value=scalar(.3);cosine=scalar(.5403023058681398)
    sine=scalar(.8414709848078965);scale=scalar(.7071067811865476)
    half=scalar(.5);c3=scalar(1/6);c4=scalar(1/24);c5=scalar(1/120);c6=scalar(1/720)
    start=scalar(.125);negative=scalar(-.125);one=scalar(1.)
    rows={}
    for fused in (False,True):
        lines=['@hc.func("c,c,c,c")',"def golden(x, y, z, zero_ct):"]
        if fused:
            # These are products of the original public weights, not new weights.
            a=repr(.2*.5403023058681398);b=repr(.2*.8414709848078965)
            lines+=["    swap = -(x.rotate(1) * mask0) + x.rotate(2).rotate(1) * mask1",
                    "    q = x * "+a+" + swap * "+b]
        else:
            lines+=["    projected = x * "+projection,
                    "    swap = -(projected.rotate(1) * mask0) + projected.rotate(2).rotate(1) * mask1",
                    "    q = projected * "+cosine+" + swap * "+sine]
        lines+=["    cached_dot = q * y","    current_dot = q * q"]
        for name,operand in (("s0","cached_dot"),("s1","current_dot")):
            lines+=["    "+name+"_pair = "+operand+" + "+operand+".rotate(1)",
                    "    "+name+" = ("+name+"_pair + "+name+"_pair.rotate(2)) * "+scale,
                    "    "+name+"_2 = "+name+" * "+name,
                    "    "+name+"_4 = "+name+"_2 * "+name+"_2",
                    "    "+name+"_low = "+name+"_2 * ("+name+" * "+c3+" + "+half+")",
                    "    "+name+"_high = "+name+"_4 * ("+name+"_2 * "+c6+" + "+name+" * "+c5+" + "+c4+")",
                    "    e"+name[-1]+" = "+name+"_high + "+name+"_low + "+name+" + "+one]
        lines+=["    denominator = e0 + e1","    residual = denominator * "+negative+" + "+one,
                "    residual2 = residual * residual","    residual4 = residual2 * residual2",
                "    inverse01 = (residual + "+one+") * (residual2 + "+one+")",
                "    inverse = inverse01 * (residual4 + "+one+")"]
        if fused:
            lines+=["    cached_value = z * "+repr(.3*.125),
                    "    current_value = x * "+repr(.3*.3*.125)]
        else:
            lines+=["    cached_value = (z * "+value+") * "+start,
                    "    current_value = ((x * "+value+") * "+value+") * "+start]
        lines+=["    numerator = e0 * cached_value + e1 * current_value",
                "    prediction = numerator * inverse",
                "    keys = y + q.rotate(2)",
                "    new_value = x * "+value,
                "    values = z + new_value.rotate(2)",
                "    return [prediction, keys, values]"]
        rows["packed-balanced-fused" if fused else "packed-balanced"]= "\n".join(lines)+"\n"
    # RoPE is the same public rotation for q and current k. Their dot product
    # is the public squared projection times the input norm; the cached dot
    # uses its original zero-padded prefix. Keep a separate exact KV branch.
    a = .2 * .5403023058681398
    b = .2 * .8414709848078965
    c = .7071067811865476
    norm = (.2*.2) * (.5403023058681398**2 + .8414709848078965**2)
    lines = ['@hc.func("c,c,c,c")', "def golden(x, y, z, zero_ct):",
             "    cached_q = x * "+repr(a*c)+" - x.rotate(1) * "+repr(b*c)+" + x.rotate(2).rotate(1) * "+repr(b*c),
             "    cached_dot = cached_q * y",
             "    current_dot = (x * "+repr(norm*c)+") * x"]
    for name, operand in (("s0", "cached_dot"), ("s1", "current_dot")):
        lines += ["    "+name+"_pair = "+operand+" + "+operand+".rotate(1)",
                  "    "+name+" = "+name+"_pair + "+name+"_pair.rotate(2)",
                  "    "+name+"_2 = "+name+" * "+name,
                  "    "+name+"_4 = "+name+"_2 * "+name+"_2",
                  "    "+name+"_low = "+name+"_2 * ("+name+" * "+c3+" + "+half+")",
                  "    "+name+"_high = "+name+"_4 * ("+name+"_2 * "+c6+" + "+name+" * "+c5+" + "+c4+")",
                  "    e"+name[-1]+" = "+name+"_high + "+name+"_low + "+name+" + "+one]
    lines += ["    denominator = e0 + e1", "    residual = denominator * "+negative+" + "+one,
              "    residual2 = residual * residual", "    residual4 = residual2 * residual2",
              "    inverse01 = (residual + "+one+") * (residual2 + "+one+")",
              "    inverse = inverse01 * (residual4 + "+one+")",
              "    cached_value = z * "+repr(.3*.125),
              "    current_value = x * "+repr(.3*.3*.125),
              "    numerator = e0 * cached_value + e1 * current_value",
              "    prediction = numerator * inverse",
              "    swap = -(x.rotate(1) * mask0) + x.rotate(2).rotate(1) * mask1",
              "    q = x * "+repr(a)+" + swap * "+repr(b),
              "    keys = y + q.rotate(2)",
              "    new_value = x * "+value,
              "    values = z + new_value.rotate(2)",
              "    return [prediction, keys, values]"]
    rows["packed-early-score"] = "\n".join(lines)+"\n"
    return rows

if __name__=="__main__":
    raise SystemExit(main(expected_model_sha256=MODEL_SHA256,make_variants=variants,probe_kind="attention"))
