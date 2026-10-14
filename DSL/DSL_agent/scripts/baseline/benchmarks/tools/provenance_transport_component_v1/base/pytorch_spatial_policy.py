"""Explicit spatial intake identities; private transport formats are never submission formats."""
from pytorch_source_package_policy import SOURCE_FORMAT as SOURCE_PACKAGE_FORMAT
SOURCE_FORMAT = "poseidon-trusted-pytorch-v27"
PACKAGE_FORMAT = "poseidon-trusted-pytorch-provenance-v27"
REFERENCE_FORMAT = "poseidon-trusted-pytorch-reference-v27"
PUBLIC_FORMAT = "poseidon-trusted-pytorch-spatial-public-v1"
PUBLIC_RECORD_FORMAT = "poseidon-public-tensor-spatial-source-capture-v1"
CORE_FORMAT = "poseidon-trusted-pytorch-spatial-core-v1"
CORE_PACKAGE_FORMAT = "poseidon-spatial-core-provenance-v1"
CORE_REFERENCE_FORMAT = "poseidon-spatial-core-reference-v1"
PUBLIC_PACKAGE_FORMAT = "poseidon-spatial-public-provenance-v1"
PUBLIC_REFERENCE_FORMAT = "poseidon-spatial-public-reference-v1"
EXPRESSION_FORMAT = "poseidon-public-expression-spatial-source-capture-v1"
CAPTURE_FORMAT = "poseidon-trusted-spatial-capture-v1"
POLICY = "poseidon-pytorch-static-spatial-v1"
INTERNAL_FORMATS = (PUBLIC_FORMAT, CORE_FORMAT)
def supports_input_call(spec):
    return type(spec) is dict and spec.get("format") in ("poseidon-trusted-pytorch-v26", SOURCE_FORMAT, "poseidon-trusted-pytorch-v28", SOURCE_PACKAGE_FORMAT, "poseidon-trusted-pytorch-v30","poseidon-trusted-pytorch-v31","poseidon-trusted-pytorch-v32","poseidon-trusted-pytorch-v33","poseidon-trusted-pytorch-v34","poseidon-trusted-pytorch-v35")

def supports_spatial(spec):
    return type(spec) is dict and spec.get("format") in (SOURCE_FORMAT, "poseidon-trusted-pytorch-v28", SOURCE_PACKAGE_FORMAT, "poseidon-trusted-pytorch-v30","poseidon-trusted-pytorch-v31","poseidon-trusted-pytorch-v32","poseidon-trusted-pytorch-v33","poseidon-trusted-pytorch-v34","poseidon-trusted-pytorch-v35")
