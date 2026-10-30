"""Accumulation and the exact R4 SumSlots schedule."""

from contextvars import ContextVar
from functools import wraps


_current = ContextVar('pt_agent_public_vectors', default=None)


def scope(function):
    """Nested transformer kernels share only their current call's public data."""
    @wraps(function)
    def scoped(backend, *args, **kwargs):
        current = _current.get()
        if current is not None and current[0] is backend:
            return function(backend, *args, **kwargs)
        token = _current.set((backend, {}, {}))
        try:
            return function(backend, *args, **kwargs)
        finally:
            _current.reset(token)
    return scoped


def memoize(backend, key, factory, *, owners=()):
    current = _current.get()
    if current is None or current[0] is not backend:
        return factory()
    _, values, references = current
    for owner in owners:
        references[id(owner)] = owner
    if key not in values:
        values[key] = factory()
    return values[key]


def vector(backend, key, factory, *, owners=()):
    current = _current.get()
    if current is None or current[0] is not backend:
        # Existing single-call CNN kernels retain their streaming allocation.
        return factory()
    def readonly():
        value = factory()
        value.flags.writeable = False
        return value
    return memoize(backend, key, readonly, owners=owners)


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
