"""Earth management using the compiler's consumed-level convention."""

from .greedy import RecordingBackend, manage
from .check import ManagementError, check
from .earth_emit import emit, emit_raw
from .mlir_read import read, load_graph
