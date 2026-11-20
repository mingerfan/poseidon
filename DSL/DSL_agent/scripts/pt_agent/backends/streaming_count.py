"""Count operations without owning readonly operands through identity caching."""

import numpy as np

from .count import CountBackend


class StreamingCountBackend(CountBackend):
    def const(self, vector):
        if isinstance(vector,np.ndarray) and not vector.flags.writeable:
            vector=np.array(vector,copy=True)
        return self._const(vector,keep_data=False)
