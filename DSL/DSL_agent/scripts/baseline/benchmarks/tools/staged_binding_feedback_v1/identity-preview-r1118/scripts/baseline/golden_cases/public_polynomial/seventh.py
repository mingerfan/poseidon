def fint(i):
    return int(np.floor(i))

def GenPoly(treeStr, coeffStr, length, scale=1.0):
    tree = [token.strip().split(' ') for token in treeStr]
    tree = [[int(istr) for istr in istrl] for istrl in tree]
    coeff = [float(token.strip()) / scale for token in coeffStr]
    cheby_mish = np.polynomial.Chebyshev(np.array(coeff, dtype=np.double))
    newtree = [[num for num in arr] for arr in tree]
    newtree[0][0] = cheby_mish
    for i, cousins in enumerate(tree):
        new_index = []
        for j, divisor in enumerate(cousins):
            if divisor <= 0:
                pass
            else:
                divisor = np.polynomial.Chebyshev([0] * divisor + [1])
                newtree[i + 1][2 * j + 1] = newtree[i][j] // divisor
                newtree[i + 1][2 * j] = newtree[i][j] % divisor
    calc_order = [(i, j, divisor, newtree[i][j]) for i, cousins in enumerate(tree) for j, divisor in enumerate(cousins) if divisor >= 0]
    calc_order = sorted(calc_order, reverse=True, key=lambda x: x[0])

    def polynomial(x):
        babyTs = []
        giantTs = {0: 1, 1: x}
        tmpTs = {}
        for i in range(1, fint(np.log2(length))):
            idx = pow(2, i)
            pre_idx = pow(2, i - 1)
            giantTs[idx] = 2 * giantTs[pre_idx] * giantTs[pre_idx] + -1
        babyTs = [x]
        for i in range(1, fint(np.log2(length))):
            idx = pow(2, i)
            babyAdd = [2 * poly * giantTs[idx] for poly in babyTs]
            sdfs = [new + old for new, old in zip(babyAdd, reversed(babyTs))]
            babyAdd = [new - old for new, old in zip(babyAdd, reversed(babyTs))]
            babyTs = babyTs + babyAdd
        tmpPoly = {}
        for i, j, deg, leaf in calc_order:
            if deg == 0:
                poly = 0
                for k in range(length // 2):
                    if len(leaf.coef) > 2 * k + 1:
                        poly += leaf.coef[2 * k + 1] * babyTs[k]
                tmpPoly[i, j] = poly
            else:
                if not deg in giantTs:
                    giantTs[deg] = 2 * giantTs[deg // 2] * giantTs[deg // 2] + -1
                tmpPoly[i, j] = tmpPoly[i + 1, 2 * j + 1] * giantTs[deg] + tmpPoly[i + 1, 2 * j]
        return tmpPoly[0, 0]
    return polynomial

@hc.func("c")
def golden(x):
    fn = GenPoly(["0"], ["0","0.25","0","0.5","0","0.25","0","0.125"], 8, scale=2)
    return fn(x)
