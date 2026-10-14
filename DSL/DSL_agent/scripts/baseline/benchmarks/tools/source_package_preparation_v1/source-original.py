"""Static source package preparation with explicit files and restricted imports."""
import ast
import errno
import hashlib
import io
import json
import keyword
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tokenize

FORMAT = "poseidon-pytorch-source-package-v1"
MAX_FILES = 8
MAX_BYTES = 65536
MAX_DEPTH = 8
MAX_AST_NODES = 32768
EXTERNAL_ROOTS = ("torch", "numpy", "math", "operator", "typing", "collections",
                  "functools", "itertools", "dataclasses", "enum", "abc", "numbers", "copy")
FORBIDDEN_ROOTS = frozenset(("os", "sys", "subprocess", "socket", "http", "urllib",
    "requests", "importlib", "builtins", "pathlib", "io", "pickle", "shutil",
    "tempfile", "multiprocessing", "ctypes", "cffi"))
DYNAMIC_CALLS = frozenset(("__import__", "eval", "exec", "compile", "open", "input"))
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class SourcePackageError(ValueError):
    def __init__(self, reason, *, file=None, line=None, column=None, context=None):
        super().__init__(reason)
        self.reason = reason
        self.location = None if file is None else dict(kind="source", file=file, line=line, column=column)
        self.context = {} if context is None else dict(context)


def check(condition, reason, **details):
    if not condition:
        raise SourcePackageError(reason, **details)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def identifier(value):
    return type(value) is str and 1 <= len(value) <= 64 and IDENTIFIER.fullmatch(value) is not None and not keyword.iskeyword(value)


def relative_path(value):
    check(type(value) is str and 0 < len(value) <= 256 and "\\" not in value,
          "source_path_invalid")
    path = PurePosixPath(value)
    check(not path.is_absolute() and path.as_posix() == value and
          1 <= len(path.parts) <= MAX_DEPTH and path.suffix == ".py", "source_path_invalid")
    check(all(identifier(part) for part in path.parts[:-1]), "source_path_invalid")
    check(identifier(path.stem), "source_path_invalid")
    check(not any(part.startswith("__") for part in path.parts[:-1]) and
          (not path.stem.startswith("__") or path.stem == "__init__"), "source_path_reserved")
    return path


def roots(values, reason):
    check(type(values) in (tuple, list, set, frozenset) and
          all(identifier(value) for value in values), reason)
    return frozenset(values)


def declarations(spec, *, reserved_module_roots=(), external_roots=EXTERNAL_ROOTS):
    check(type(spec) is dict and set(spec) == {"format", "files", "entry"} and
          spec["format"] == FORMAT, "source_package_fields")
    check(type(spec["files"]) is dict and 1 <= len(spec["files"]) <= MAX_FILES,
          "source_file_capacity")
    external = roots(external_roots, "source_external_roots_invalid")
    check(not external.intersection(FORBIDDEN_ROOTS), "source_external_roots_forbidden")
    extra_reserved = roots(reserved_module_roots, "source_reserved_roots_invalid")
    reserved = frozenset(sys.stdlib_module_names) | external | extra_reserved | {
        "torch", "numpy", "hecate", "poly", "mlir", "einops"}
    modules = {}
    paths = {}
    for name, expected in spec["files"].items():
        path = relative_path(name)
        check(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
              "source_hash_invalid", file=name)
        parts = list(path.parts[:-1]) + [path.stem]
        is_package = path.stem == "__init__"
        if is_package:
            check(len(parts) > 1, "source_root_initializer_unbound", file=name)
            parts.pop()
        check(parts[0] not in reserved, "source_module_reserved", file=name)
        module = ".".join(parts)
        check(module not in modules, "source_module_collision", file=name)
        modules[module] = dict(file=name, kind="package" if is_package else "module")
        paths[name] = path
    for module, item in modules.items():
        parts = module.split(".")
        for count in range(1, len(parts)):
            parent = ".".join(parts[:count])
            check(parent in modules and modules[parent]["kind"] == "package",
                  "source_package_parent_missing", file=item["file"])
    check(type(spec["entry"]) is str and "." in spec["entry"], "source_entry_invalid")
    module, factory = spec["entry"].rsplit(".", 1)
    check(all(identifier(part) for part in module.split(".")) and identifier(factory) and
          not factory.startswith("__"), "source_entry_invalid")
    check(module in modules, "source_entry_module_missing")
    return dict(modules=dict(sorted(modules.items())), entry=dict(module=module, factory=factory),
                external_roots=sorted(external), reserved_roots_sha256=digest(sorted(reserved)))


def source_text(raw, name):
    check(type(raw) is bytes, "source_bytes_invalid", file=name)
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        check(encoding in ("utf-8", "utf-8-sig"), "source_encoding_unsupported", file=name)
        return raw.decode(encoding)
    except (UnicodeError, SyntaxError):
        raise SourcePackageError("source_encoding_invalid", file=name) from None


def imports_for(tree, module, kind, info, name):
    result = []
    package = module if kind == "package" else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            check(node.func.id not in DYNAMIC_CALLS, "source_dynamic_execution",
                  file=name, line=node.lineno, column=node.col_offset)
        if isinstance(node, ast.Import):
            targets = [(alias.name, dict(name=alias.name, asname=alias.asname)) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            target = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                count = len(parts) - node.level + 1
                check(count >= 1, "source_relative_import_escape", file=name, line=node.lineno,
                      column=node.col_offset)
                target = ".".join(parts[:count] + ([target] if target else []))
            targets = [(target, dict(name=alias.name, asname=alias.asname)) for alias in node.names]
        else:
            continue
        for target, alias in targets:
            check(target and all(identifier(part) for part in target.split(".")),
                  "source_import_invalid", file=name, line=node.lineno, column=node.col_offset)
            top = target.split(".")[0]
            local = top in {key.split(".")[0] for key in info["modules"]}
            if local:
                check(target in info["modules"], "source_import_not_declared",
                      file=name, line=node.lineno, column=node.col_offset)
                category = "local"
            else:
                check(top in info["external_roots"], "source_import_not_registered",
                      file=name, line=node.lineno, column=node.col_offset)
                category = "registered_external"
            result.append(dict(file=name, line=node.lineno, column=node.col_offset,
                               module=target, alias=alias, category=category))
    return result


def validate(spec, raw_sources, *, reserved_module_roots=(), external_roots=EXTERNAL_ROOTS):
    info = declarations(spec, reserved_module_roots=reserved_module_roots, external_roots=external_roots)
    check(type(raw_sources) is dict and set(raw_sources) == set(spec["files"]), "source_inventory_mismatch")
    files = {}
    imports = []
    total_bytes = total_nodes = 0
    for module, item in info["modules"].items():
        name = item["file"]
        raw = raw_sources[name]
        check(type(raw) is bytes, "source_bytes_invalid", file=name)
        total_bytes += len(raw)
        check(total_bytes <= MAX_BYTES, "source_byte_capacity",
              context=dict(observed=total_bytes, limit=MAX_BYTES))
        check(hashlib.sha256(raw).hexdigest() == spec["files"][name], "source_hash_mismatch", file=name)
        text = source_text(raw, name)
        try:
            tree = ast.parse(text, filename=name)
        except (SyntaxError, ValueError, RecursionError):
            raise SourcePackageError("source_syntax_invalid", file=name) from None
        total_nodes += sum(1 for _ in ast.walk(tree))
        check(total_nodes <= MAX_AST_NODES, "source_ast_capacity",
              context=dict(observed=total_nodes, limit=MAX_AST_NODES))
        imports.extend(imports_for(tree, module, item["kind"], info, name))
        files[name] = dict(sha256=spec["files"][name], bytes=len(raw), module=module, kind=item["kind"])
    record = dict(format=FORMAT, source_files=dict(sorted(files.items())), modules=info["modules"],
                  entry=info["entry"], external_roots=info["external_roots"],
                  reserved_roots_sha256=info["reserved_roots_sha256"],
                  imports=sorted(imports, key=lambda row: (row["file"], row["line"], row["column"],
                                                          row["module"], row["alias"]["name"])),
                  counts=dict(files=len(files), bytes=total_bytes, ast_nodes=total_nodes),
                  limits=dict(files=MAX_FILES, bytes=MAX_BYTES, path_depth=MAX_DEPTH, ast_nodes=MAX_AST_NODES),
                  status="prepared_not_executed", factory_execution="not_run",
                  graph_capture="not_run", backend_execution="not_run")
    record["binding"] = digest(record)
    return record


def _root_descriptor(root):
    check(type(root) is str or isinstance(root, Path), "source_root_invalid")
    check(bool(os.fspath(root)), "source_root_invalid")
    absolute = Path(os.path.abspath(os.fspath(root)))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError:
        os.close(descriptor)
        raise SourcePackageError("source_root_unavailable") from None


def _read_relative(root_descriptor, path, limit):
    directories = []
    descriptor = root_descriptor
    file_descriptor = None
    try:
        for part in path.parts[:-1]:
            descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                 dir_fd=descriptor)
            directories.append(descriptor)
        file_descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                  dir_fd=descriptor)
        before = os.fstat(file_descriptor)
        check(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "source_file_type", file=path.as_posix())
        check(0 <= before.st_size <= limit, "source_byte_capacity",
              context=dict(observed=before.st_size, limit=limit))
        data = bytearray()
        while len(data) <= limit:
            chunk = os.read(file_descriptor, min(65536, limit + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(file_descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        check(len(data) == before.st_size and all(getattr(before, key) == getattr(after, key) for key in fields),
              "source_changed_during_read", file=path.as_posix())
        return bytes(data)
    except OSError as error:
        reason = "source_path_link" if error.errno in (errno.ELOOP, errno.ENOTDIR) else "source_file_unavailable"
        raise SourcePackageError(reason, file=path.as_posix()) from None
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(directories):
            os.close(descriptor)


def prepare(root, spec, *, reserved_module_roots=(), external_roots=EXTERNAL_ROOTS):
    declarations(spec, reserved_module_roots=reserved_module_roots, external_roots=external_roots)
    descriptor = _root_descriptor(root)
    try:
        files = {}
        total = 0
        for name in sorted(spec["files"]):
            raw = _read_relative(descriptor, relative_path(name), MAX_BYTES - total)
            total += len(raw)
            files[name] = raw
    finally:
        os.close(descriptor)
    record = validate(spec, files, reserved_module_roots=reserved_module_roots, external_roots=external_roots)
    return record, files
