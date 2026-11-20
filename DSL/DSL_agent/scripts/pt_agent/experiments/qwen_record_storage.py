"""Detached, exact storage for private refresh-record metadata.

This helper has no runner or admission. It stores original call values without
owning any backend, expression, parameter array or model. A future private
factory may install it before the first input; existing sources stay unchanged.
"""
from array import array
from collections.abc import Sequence
import operator
import sys

MAX_ID = 2**63-1


def _node(value):
    if type(value) is not int or not 0 <= value <= MAX_ID:
        raise ValueError('A node identifier must be a nonnegative signed64 integer')
    return value


class PackedCalls(Sequence):
    """Append/read/clear sequence of exact (operands, result, extra) tuples.

    One signed64 stream contains arity, result, operands and extras. Unsigned64
    offsets delimit rows. Operand None alone is encoded as-1. Indexing returns
    original Python tuple values; slices explicitly materialize ordinary lists.
    """
    __slots__ = ('_offsets', '_data')

    def __init__(self):
        self.clear()
        if self._offsets.itemsize != 8 or self._data.itemsize != 8:
            raise RuntimeError('Packed call storage requires64bit integer arrays')

    def append(self, row):
        if type(row) is not tuple or len(row) != 3:
            raise ValueError('A call is an operands/result/extra tuple')
        operands, result, extra = row
        if type(operands) is not tuple or type(extra) is not tuple:
            raise ValueError('Call operands and extras must remain tuples')
        encoded = array('q', (len(operands), _node(result)))
        encoded.extend(-1 if value is None else _node(value) for value in operands)
        encoded.extend(_node(value) for value in extra)
        start = len(self._data)
        rows = len(self._offsets)
        try:
            self._data.extend(encoded)
            self._offsets.append(len(self._data))
        except BaseException:
            del self._data[start:]
            del self._offsets[rows:]
            raise

    def __len__(self):
        return len(self._offsets)-1

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[index] for index in range(*key.indices(len(self)))]
        index = operator.index(key)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError('Packed call index out of range')
        start, end = self._offsets[index], self._offsets[index+1]
        arity, result = self._data[start], self._data[start+1]
        operands = tuple(None if value == -1 else value for value in self._data[start+2:start+2+arity])
        extra = tuple(self._data[start+2+arity:end])
        return operands, result, extra

    def __eq__(self, other):
        if not isinstance(other, (PackedCalls, list)):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    def __ne__(self, other):
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def clear(self):
        self._offsets = array('Q', (0,))
        self._data = array('q')

    def storage_bytes(self):
        """Owned object and allocated array capacity, including allocator slack."""
        return sys.getsizeof(self)+sys.getsizeof(self._offsets)+sys.getsizeof(self._data)

    def payload_bytes(self):
        return self._offsets.itemsize*len(self._offsets)+self._data.itemsize*len(self._data)


class InternedStages(list):
    """Original list order/values, sharing equal immutable statement-stage pairs."""
    __slots__ = ('_pairs',)

    def __init__(self):
        super().__init__()
        self._pairs = {}

    def append(self, pair):
        if type(pair) is not tuple or len(pair) != 2 or any(type(v) is not int for v in pair):
            raise ValueError('A recorded stage is a pair of Python integers')
        super().append(self._pairs.setdefault(pair, pair))

    def clear(self):
        super().clear()
        self._pairs.clear()

    def storage_bytes(self):
        return sys.getsizeof(self)+sys.getsizeof(self._pairs)+sum(sys.getsizeof(pair) for pair in self._pairs)


def install_empty_record_storage(recorder):
    """Future factory hook, explicit fresh-record admission; no arithmetic."""
    if (recorder.policy != 'record' or recorder.call_count != 0
            or type(recorder.calls) is not list or recorder.calls
            or type(recorder.node_stage) is not list or recorder.node_stage
            or recorder.output_nodes is not None):
        raise ValueError('Install exact storage before any recording input or call')
    recorder.calls, recorder.node_stage = PackedCalls(), InternedStages()
    return recorder
