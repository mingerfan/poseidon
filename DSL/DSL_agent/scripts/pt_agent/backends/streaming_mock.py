"""Independent diagnostic Mock arithmetic with ephemeral plaintext operands."""

import numpy as np

from .mock import MockBackend


class StreamingMockBackend(MockBackend):
    """Keep constant names in the backend and values only in live operands.

    Copy read-only inputs into a writable temporary so the base identity cache
    cannot retain the caller's array.  No original weight array is modified.
    """
    _keep_const_data = False

    def __init__(self, slots, batch=None):
        super().__init__(slots, batch=batch, stream_constants=False)

    def const(self, vec):
        if isinstance(vec, np.ndarray) and not vec.flags.writeable:
            vec = np.array(vec, copy=True)
        return self._const(vec, keep_data=True)
