"""Accumulation and the exact R4 SumSlots schedule."""


def accumulate(acc, term):
    return term if acc is None else acc+term


def sum_slots(b, A, M: int, stride: int):
    if M == 1:
        return A
    L = M.bit_length()-1
    B = [A]
    for j in range(1, L+1):
        B.append(accumulate(B[j-1], B[j-1].rotate(2**(j-1)*stride)))
    C = B[L]
    for j in range(L):
        if (M//2**j) % 2 == 1:
            C = accumulate(C, B[j].rotate((M//2**(j+1))*2**(j+1)*stride))
    return C
