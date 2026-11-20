"""Optional intake timing contexts, independent of result and configuration state."""

from contextlib import nullcontext


def observe(observer, stage_id, **metadata):
    """Use the supplied observer, or leave the enclosed operation unchanged."""
    return nullcontext() if observer is None else observer.span(stage_id, **metadata)
