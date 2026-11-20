"""Read-only source closure for trusted capture workers; never import model code."""
import ast
from functools import lru_cache
from pathlib import Path
import re


@lru_cache(maxsize=256)
def dependency_names(raw):
    """Cache immutable syntax only, keyed by the complete source bytes."""
    candidates = set()
    for node in ast.walk(ast.parse(raw.decode("utf-8"))):
        if isinstance(node, ast.Import):
            candidates.update(alias.name.split(".")[0]+".py" for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            candidates.add(node.module.split(".")[0]+".py")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "with_name" and len(node.args) == 1
              and isinstance(node.args[0], ast.Constant) and type(node.args[0].value) is str):
            adjacent = node.args[0].value
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*\.(py|json)", adjacent):
                candidates.add(adjacent)
    return frozenset(candidates)


def implementation_paths_at(local, retained, initial):
    """Follow trusted imports and explicit adjacent data, including lazy imports."""
    local, retained = Path(local), Path(retained)
    pending, result = list(initial), {}
    while pending:
        name = pending.pop()
        if name in result:
            continue
        if (type(name) is not str or Path(name).name != name
                or Path(name).suffix not in (".py", ".json")):
            raise ValueError("Trusted implementation filename")
        path = local/name if (local/name).is_file() else retained/name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Trusted implementation file missing or indirect")
        if len(result) >= 4096:
            raise ValueError("Trusted implementation file budget")
        result[name] = path
        if path.suffix == ".py":
            candidates = dependency_names(path.read_bytes())
            pending.extend(sorted(n for n in candidates if n not in result
                                  and ((local/n).is_file() or (retained/n).is_file())))
    return dict(sorted(result.items()))
