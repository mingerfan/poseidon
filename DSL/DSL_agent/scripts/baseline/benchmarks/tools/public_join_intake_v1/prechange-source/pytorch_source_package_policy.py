"""Source organization identity; v29 inherits v28 tensor computation rules."""
SOURCE_FORMAT = "poseidon-trusted-pytorch-v29"
PACKAGE_FORMAT = "poseidon-trusted-pytorch-provenance-v29"
REFERENCE_FORMAT = "poseidon-trusted-pytorch-reference-v29"
CAPTURE_FORMAT = "poseidon-public-expression-source-capture-v6"
POLICY = "poseidon-pytorch-explicit-source-package-v1"


def selected(spec):
    return type(spec) is dict and spec.get("format") in (SOURCE_FORMAT, "poseidon-trusted-pytorch-v30","poseidon-trusted-pytorch-v31","poseidon-trusted-pytorch-v32","poseidon-trusted-pytorch-v33")


def envelope(spec):
    from pytorch_source_package import FORMAT
    return dict(format=FORMAT, files=spec["files"], entry=spec["entry"])


def raw_sources(files):
    from pytorch_source_package import check
    check(type(files) is dict, "source_inventory_mismatch")
    check(all(type(value) in (str, bytes) for value in files.values()), "source_bytes_invalid")
    return {name: value.encode("utf-8") if type(value) is str else value
            for name, value in files.items()}


def mount_directories(spec):
    from pathlib import PurePosixPath
    from pytorch_source_package import relative_path
    parents = {parent.as_posix()
               for name in spec["files"]
               for parent in relative_path(name).parents
               if parent != PurePosixPath(".")}
    arguments = []
    for name in sorted(parents, key=lambda value: (value.count("/"), value)):
        arguments.extend(("--dir", "/model/" + name))
    return arguments
