"""Exact Chebyshev identities for a bounded manual feasibility probe.

No coefficient projection/truncation and no bootstrap or parameter changes.
This is an explicit option of the trusted rule generator, not an Agent rewrite.
"""
def balanced(x,coeff,zero):
    terms={0:zero+1.0,1:x}
    def term(n):
        if n not in terms:
            k=n//2
            if n%2:
                product=term(k)*term(k+1)
                terms[n]=product+product-x
            else:
                square=term(k)*term(k)
                terms[n]=square+square-1.0
        return terms[n]
    result=zero
    for degree,c in enumerate(coeff):
        # Keep every nonzero coefficient, including coefficients smaller than
        # CKKS encoding precision. No threshold is used to erase terms.
        if float(c)==0.0:continue
        result=result+term(degree)*float(c)
    return result
