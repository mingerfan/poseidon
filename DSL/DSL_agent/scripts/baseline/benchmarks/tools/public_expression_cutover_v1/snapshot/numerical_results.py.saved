"""Backend-independent numerical acceptance and strict JSON reports."""
import json

def require(condition, reason):
    if not condition:
        raise ValueError(reason)

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def compare(actual, reference, atol, rtol):
    import numpy as np
    actual, reference = np.asarray(actual), np.asarray(reference)
    require(actual.shape == reference.shape and actual.ndim == 2 and actual.size > 0, "Output shape mismatch")
    require(np.isfinite(actual).all() and np.isfinite(reference).all(), "Non-finite numerical result")
    error = np.abs(actual - reference)
    nonzero = reference != 0
    relative = np.full(reference.shape, np.nan)
    relative[nonzero] = error[nonzero] / np.abs(reference[nonzero])
    norm = float(np.linalg.norm(actual) * np.linalg.norm(reference))
    return dict(passed=bool(np.all(error <= atol + rtol * np.abs(reference))), atol=atol, rtol=rtol,
                compared_values=int(actual.size), mae=float(error.mean()), max_absolute_error=float(error.max()),
                max_nonzero_reference_relative_error=float(relative[nonzero].max()) if nonzero.any() else None,
                nonzero_reference_count=int(nonzero.sum()),
                cosine_similarity=float(np.vdot(actual.ravel(), reference.ravel()) / norm) if norm else None,
                actual=actual.tolist(), reference=reference.tolist(), absolute_error=error.tolist(),
                relative_error=[[float(x) if np.isfinite(x) else None for x in row] for row in relative],
                elementwise_pass=(error <= atol + rtol * np.abs(reference)).tolist())
