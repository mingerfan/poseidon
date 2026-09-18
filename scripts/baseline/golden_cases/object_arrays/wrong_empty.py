def fint(i):
    return int(np.floor(i))

def roll(A, i):
    return A.rotate(-i)

def SumSlots(A, m, p):
    B = np.full((fint(np.log2(m)) + 1,), Empty(), dtype=object)
    B[0] = A
    for j in range(1, fint(np.log2(m)) + 1):
        B[j] = B[j - 1] + roll(B[j - 1], -pow(2, j - 1) * p)
    C = B[fint(np.log2(m))]
    for j in range(0, fint(np.log2(m))):
        if m // pow(2, j) % 2 == 1:
            C = C + roll(B[j], -(m // pow(2, j + 1)) * pow(2, j + 1) * p)
    return C

@hc.func('c')
def golden(x):
    a = np.full((2,), Empty(), dtype=object)
    a[0] -= SumSlots(x, 1, 1)
    b = a[:1]
    c = b.copy()
    b[0] = -b[0]*c0
    return b[0]+c[0]+c1
