"""Grouped exact CSE keys with borrowed operands and global insertion order.

The supported key domain is the original CostDag (kind, operands, step) tuple.
Only its outer tuple is factored into groups. Operand tuples stay unchanged.
"""
from array import array
from collections.abc import MutableMapping


class _Bucket(dict):
    __slots__ = ('groupid',)

    def __init__(self, groupid):
        super().__init__()
        self.groupid = groupid


class GroupedIds(MutableMapping):
    """Original keys/values, including insertion order and delete/reinsert order.

    Structural changes invalidate iterators. Updating an existing value keeps
    its position. Delete scans order metadata; the production CSE only clears.
    """
    def __init__(self):
        self._groups = {}
        self._group_keys = []
        self._order_groups = array('Q')
        self._order_operands = []
        self._length = 0
        self._version = 0

    @staticmethod
    def _parts(key):
        kind, operands, step = key
        return (kind, step), operands

    def __getitem__(self, key):
        group, operands = self._parts(key)
        try:
            return self._groups[group][operands]
        except KeyError:
            raise KeyError(key) from None

    def __contains__(self, key):
        group, operands = self._parts(key)
        bucket = self._groups.get(group)
        return bucket is not None and operands in bucket

    def __setitem__(self, key, value):
        group, operands = self._parts(key)
        bucket = self._groups.get(group)
        if bucket is None:
            bucket = _Bucket(len(self._group_keys))
            self._groups[group] = bucket
            self._group_keys.append(group)
        if operands in bucket:
            bucket[operands] = value
            return
        bucket[operands] = value
        self._order_groups.append(bucket.groupid)
        self._order_operands.append(operands)
        self._length += 1
        self._version += 1

    def __delitem__(self, key):
        group, operands = self._parts(key)
        try:
            bucket = self._groups[group]
            del bucket[operands]
        except KeyError:
            raise KeyError(key) from None
        for index, saved in enumerate(self._order_operands):
            if (saved is not None and self._order_groups[index] == bucket.groupid
                    and (saved is operands or saved == operands)):
                self._order_operands[index] = None
                break
        self._length -= 1
        self._version += 1

    def __iter__(self):
        version = self._version
        def keys():
            for index in range(len(self._order_operands)):
                if version != self._version:
                    raise RuntimeError('GroupedIds changed during iteration')
                operands = self._order_operands[index]
                if operands is not None:
                    kind, step = self._group_keys[self._order_groups[index]]
                    yield kind, operands, step
            if version != self._version:
                raise RuntimeError('GroupedIds changed during iteration')
        return keys()

    def __len__(self):
        return self._length

    def clear(self):
        for bucket in self._groups.values():
            bucket.clear()
        self._groups.clear()
        self._group_keys.clear()
        del self._order_groups[:]
        self._order_operands.clear()
        self._length = 0
        self._version += 1
