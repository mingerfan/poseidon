"""Explicit logical axis-view normalization after trusted PyTorch capture."""
SOURCE_FORMAT="poseidon-trusted-pytorch-v35"
PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v35"
REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v35"
CAPTURE_FORMAT="poseidon-public-expression-source-capture-v12"
POLICY="poseidon-static-logical-views-v1"

def selected(spec):
    return type(spec) is dict and spec.get("format") == SOURCE_FORMAT
