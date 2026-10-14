"""Versioned application identities over the conservative local call/import closure.

Legacy Poseidon package/backend identity is unchanged. A stage binds whole local
modules, including imports inside functions, because Python calls and callbacks
are dynamic. Shared request reconstruction therefore still couples some prompt
rules to compilation/validation. This is not a claim of perfect stage separation.
No module is imported or executed while discovering its source dependencies.
"""
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

FORMAT = "poseidon-application-identities-v1"
STAGE_FORMAT = "poseidon-application-stage-identity-v1"
POLICY = "conservative-local-module-closure-v1"
STAGES = ("generation", "compilation", "execution", "validation")
DEFAULT_COMPATIBILITY = ("compilation", "execution", "validation")
TOOL_FIELDS = {
    "generation": ("platform", "source_lock_sha256", "source_lock",
                   "compiler_profile_sha256", "operator_spec_sha256"),
    "compilation": ("platform", "source_lock_sha256", "source_lock",
                    "runtime_dacapo_tree_sha256", "compiler_sha256", "frontend_sha256",
                    "compiler_profile_sha256", "operator_spec_sha256"),
    "execution": ("platform", "source_lock_sha256", "source_lock",
                  "runtime_dacapo_tree_sha256", "api_tree_sha256", "runner_tree_sha256",
                  "executor_sha256", "poseidon_library_sha256",
                  "compiler_profile_sha256", "operator_spec_sha256"),
    "validation": ("platform", "source_lock_sha256", "source_lock",
                   "compiler_profile_sha256", "operator_spec_sha256"),
}
CACHES = {"generation": (), "compilation": ("compiler",),
          "execution": ("executor", "poseidon"), "validation": ()}


def _digest(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, OverflowError) as error:
        raise ValueError("Identity must contain bounded JSON data") from error
    return hashlib.sha256(raw).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _relative(name):
    if type(name) is not str:
        return False
    p = Path(name)
    return (bool(name) and not p.is_absolute()
            and ".." not in p.parts and "." not in p.parts
            and str(p) == name and all(part not in (".git", "__pycache__") for part in p.parts))


def _read(root, name):
    _require(_relative(name), "Identity source path")
    p = root / name
    _require(p.resolve().is_relative_to(root.resolve()) and
             not any((root / Path(*p.relative_to(root).parts[:i])).is_symlink()
                     for i in range(1, len(p.relative_to(root).parts) + 1)),
             "Identity source symlink/escape")
    _require(p.is_file() and p.stat().st_size <= 16 * 1024**2, "Identity source file/size: " + name)
    return p.read_bytes()



CATALOG = "application-source-catalog-v1.json"

def source_catalog(root):
    """Read a bounded, data-only authority; paths and loaders are never executed."""
    raw=_read(Path(root),CATALOG)
    _require(len(raw)<=32768,"Identity catalog byte budget")
    def pairs(items):
        result={}
        for key,value in items:
            _require(key not in result,"Duplicate identity catalog field")
            result[key]=value
        return result
    value=json.loads(raw,object_pairs_hook=pairs)
    _require(type(value) is dict and set(value)=={"format","roots","optional_roots","dynamic_loads"}
             and value["format"]=="poseidon-identity-source-catalog-v1","Identity catalog fields")
    def paths(items):
        _require(type(items) is list and len(items)<=32 and all(_relative(p) and p.endswith(".py") for p in items)
                 and len(items)==len(set(items)),"Identity catalog paths")
        return tuple(items)
    _require(type(value["roots"]) is dict and set(value["roots"])==set(STAGES),"Identity catalog stages")
    roots={stage:paths(value["roots"][stage]) for stage in STAGES}
    _require(all(roots.values()),"Identity catalog nonempty stage")
    optional=value["optional_roots"]
    _require(type(optional) is dict and set(optional)<=set(STAGES),"Identity optional stages")
    for stage,items in optional.items():
        _require(not set(paths(items)).intersection(roots[stage]),"Duplicate optional identity path")
    loads=value["dynamic_loads"]
    _require(type(loads) is list and len(loads)<=64,"Identity dynamic authority budget")
    seen=set()
    for entry in loads:
        _require(type(entry) is dict and set(entry)=={"module","expression","authority"},"Identity dynamic authority fields")
        module,expression,authority=(entry[k] for k in ("module","expression","authority"))
        _require(_relative(module) and module.endswith(".py") and type(expression) is str and 0<len(expression)<=1024
                 and type(authority) is str and 0<len(authority)<=256,"Identity dynamic authority")
        parsed=ast.parse(expression,mode="eval")
        _require(type(parsed.body) is ast.Call and ast.unparse(parsed.body)==expression,"Identity exact dynamic call")
        pair=(module,expression);_require(pair not in seen,"Duplicate identity dynamic authority");seen.add(pair)
    return value

def _catalog_values(value):
    return ({s:tuple(v) for s,v in value["roots"].items()},
            {s:tuple(v) for s,v in value["optional_roots"].items()},
            {(e["module"],e["expression"]):e["authority"] for e in value["dynamic_loads"]})

ROOTS,OPTIONAL_ROOTS,DYNAMIC=_catalog_values(source_catalog(Path(__file__).parent))

# Cache syntax facts only. Every identity still reads and hashes current bytes,
# resolves current local files, and checks the current dynamic-load authority.
# Cached trees, complete closures and backend identities are never retained.
from collections import OrderedDict
_SYNTAX_CACHE = OrderedDict()
_SYNTAX_CACHE_BYTES = 0
# Retain a ~640-module closure across the four stages; keep the byte ceiling.
SYNTAX_CACHE_ENTRIES = 1024
SYNTAX_CACHE_BYTES = 2 * 1024**2

def _syntax_facts(raw, name):
    global _SYNTAX_CACHE_BYTES
    key = (name, hashlib.sha256(raw).hexdigest())
    if key in _SYNTAX_CACHE:
        value, size = _SYNTAX_CACHE.pop(key)
        _SYNTAX_CACHE[key] = (value, size)
        return value
    tree = ast.parse(raw, filename=name)
    nodes = list(ast.walk(tree))
    aliases = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = (node.module + "." if node.module else "") + item.name
    modules, literals, calls = [], [], []
    for node in nodes:
        if isinstance(node, ast.Import):
            modules.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                parents = Path(name).parent.parts
                _require(node.level <= len(parents)+1, "Relative import escapes identity root")
                base = parents[:len(parents)-node.level+1]
                module = ".".join((*base,module)) if module else ".".join(base)
            modules.append(module)
            modules.extend(module+"."+a.name for a in node.names if a.name!="*")
        if (isinstance(node,ast.Constant) and type(node.value) is str
                and re.fullmatch(r"[A-Za-z0-9_./-]+\.(py|json|patch|sh|nix)",node.value)
                and _relative(node.value)):
            literals.append(node.value)
        if not isinstance(node,ast.Call):
            continue
        function = ast.unparse(node.func)
        head,*tail = function.split(".")
        resolved = ".".join((aliases.get(head,head),*tail))
        if (resolved not in ("__import__","eval","exec","importlib.import_module",
                             "importlib.util.spec_from_file_location")
                and not resolved.endswith(".exec_module")):
            continue
        literal = (node.args[0].value if resolved in ("__import__","importlib.import_module")
                   and node.args and isinstance(node.args[0],ast.Constant)
                   and type(node.args[0].value) is str else None)
        calls.append((node.lineno,literal,ast.unparse(node)))
    mounts = []
    if name.removesuffix(".py") in ("sandbox_policy", "poseidon_sandbox_policy"):
        declarations = [node for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "MODULES" for t in node.targets)]
        _require(len(declarations) == 1, "Sandbox dependency inventory")
        values = ast.literal_eval(declarations[0].value)
        _require(type(values) is tuple and bool(values) and all(_relative(v) for v in values)
                 and len(values) == len(set(values)), "Sandbox dependency inventory")
        mounts.extend(values)
    value = (tuple(modules),tuple(literals),tuple(calls),tuple(mounts))
    size = len(json.dumps(value,separators=(",",":")).encode())
    if size <= SYNTAX_CACHE_BYTES:
        while _SYNTAX_CACHE and (len(_SYNTAX_CACHE)>=SYNTAX_CACHE_ENTRIES
                                 or _SYNTAX_CACHE_BYTES+size>SYNTAX_CACHE_BYTES):
            _,(_,removed)=_SYNTAX_CACHE.popitem(last=False)
            _SYNTAX_CACHE_BYTES -= removed
        _SYNTAX_CACHE[key]=(value,size)
        _SYNTAX_CACHE_BYTES += size
    return value


def source_closure(root, roots, *, dynamic_authorities=None):
    """Discover conservative local imports plus audited non-import worker mounts.

    Unknown dynamic execution fails closed. It must acquire an explicit source
    authority before it can be used for cache/package compatibility.
    """
    root = Path(root)
    authority_map=DYNAMIC if dynamic_authorities is None else dynamic_authorities
    roots = sorted(set(roots))
    _require(roots and all(_relative(p) and p.endswith(".py") for p in roots), "Identity roots")
    pending = list(roots)
    files, edges, dynamic, external = {}, {}, {}, set()
    parsed = set()

    def local(module):
        if not module:
            return []
        stem = module.replace(".", "/")
        candidates = (stem + ".py", stem + "/__init__.py")
        return [p for p in candidates if (root / p).is_file()]

    def include(name, children):
        if name not in children:
            children.append(name)
        if name not in parsed:
            pending.append(name)

    while pending:
        name = pending.pop()
        if name in parsed:
            continue
        parsed.add(name)
        raw = _read(root, name)
        files[name] = hashlib.sha256(raw).hexdigest()
        if not name.endswith(".py"):
            continue
        modules, literals, calls, mounts = _syntax_facts(raw, name)
        children = []
        for module in modules:
            paths = local(module)
            for path in paths:
                include(path, children)
                parts = Path(path).parts
                for i in range(1, len(parts)):
                    init = str(Path(*parts[:i]) / "__init__.py")
                    if (root / init).is_file():
                        include(init, children)
            if not paths and module:
                external.add(module.split(".")[0])
        for value in literals:
            if (root/value).is_file():
                include(value,children)
        for lineno,module,expression in calls:
            if module is not None:
                paths=local(module)
                for path in paths:
                    include(path,children)
                if not paths:
                    external.add(module.split(".")[0])
                dynamic[name+":"+str(lineno)]="literal import: "+module
            else:
                authority=authority_map.get((name,expression))
                _require(authority is not None,"Unbounded dynamic dependency: "+name+": "+expression)
                dynamic[name+":"+str(lineno)]=authority
        for value in mounts:
            include(value,children)
        # Verification iterates every adapter, including dynamically selected helpers.
        if name == "upstream_candidate_helpers.py":
            for p in sorted((root / "upstream_adapters").glob("*.py")):
                relative = str(p.relative_to(root))
                # Existing verify_sources hashes tests too; they are data dependencies,
                # not executed modules. An actual import will still parse them.
                files[relative] = hashlib.sha256(_read(root, relative)).hexdigest()
                if relative not in children:
                    children.append(relative)
        if name == "python_compiler_smoke.py":
            for p in sorted((root / "golden_cases").rglob("*")):
                if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts:
                    include(str(p.relative_to(root)), children)
        edges[name] = sorted(children)
        _require(len(files) + len(pending) <= 10000, "Identity source count")
    return dict(roots=roots, sources=dict(sorted(files.items())),
                import_edges=dict(sorted(edges.items())),
                dynamic_dependencies=dict(sorted(dynamic.items())),
                external_imports=sorted(external))


def _backend_projection(backend, stage):
    _require(type(backend) is dict and backend.get("format") == "poseidon-backend-identity-v2",
             "Pinned Poseidon backend identity v2 required")
    required = {"format", "build_configuration", *set().union(*[set(v) for v in TOOL_FIELDS.values()])}
    _require(required <= set(backend), "Incomplete backend identity")
    value = {k: copy.deepcopy(backend[k]) for k in ("format", *TOOL_FIELDS[stage])}
    config = backend["build_configuration"]
    _require(type(config) is dict and type(config.get("environment_files")) is dict
             and type(config.get("cmake_cache_sha256")) is dict, "Backend build configuration")
    caches = config["cmake_cache_sha256"]
    _require({"compiler", "executor", "poseidon"} <= set(caches), "Backend build cache identity")
    value["build_configuration"] = dict(environment_files=copy.deepcopy(config["environment_files"]),
        cmake_cache_sha256={k: caches[k] for k in CACHES[stage]})
    # Future fields are conservatively bound to every stage until classified.
    value["unclassified_backend_fields"] = {k: copy.deepcopy(v) for k, v in backend.items() if k not in required}
    value["unclassified_build_fields"] = {k: copy.deepcopy(v) for k, v in config.items()
                                           if k not in ("environment_files", "cmake_cache_sha256")}
    value["unclassified_build_caches"] = {k: v for k, v in caches.items()
                                            if k not in ("compiler", "executor", "poseidon")}
    return value


def _repository_references(root, sources):
    """Bind literal launcher/configuration paths outside the baseline directory."""
    records = {}
    baseline = root / "scripts/baseline"
    for name in sources:
        if not name.endswith(".py"):
            continue
        _,literals,_,_ = _syntax_facts(_read(baseline,name),name)
        for value in literals:
            if (root/value).is_file():
                records[value]=hashlib.sha256(_read(root,value)).hexdigest()
    return records


def identity_manifest(root, backend_identity):
    """Build from current bytes and the unchanged trusted low-level identity."""
    root = Path(root)
    baseline = root / "scripts/baseline"
    catalog=source_catalog(baseline)
    stage_roots,optional_roots,authorities=_catalog_values(catalog)
    result = dict(format=FORMAT, backend=copy.deepcopy(backend_identity), policy=POLICY)
    # Public configuration/semantic tables and patches can be selected dynamically.
    # No tests, docs, run reports or benchmark tool directory are globbed here.
    assets = sorted(set([str(p.relative_to(root)) for p in baseline.glob("*.json")] +
                        [str(p.relative_to(root)) for p in baseline.glob("patches/*.patch")] +
                        [str(p.relative_to(root)) for p in (root / "src/poseidon/tools/dacapo").glob("*.json")] +
                        [str(p.relative_to(root)) for p in (root / "src/poseidon/tools/dacapo").glob("*.nix")]))
    asset_hashes = {p: hashlib.sha256(_read(root, p)).hexdigest() for p in assets}
    for stage in STAGES:
        roots=list(stage_roots[stage])+["application_identity.py","application_backend.py"]
        roots.extend(name for name in optional_roots.get(stage,()) if (baseline/name).is_file())
        closure=source_closure(baseline,roots,dynamic_authorities=authorities)
        payload = dict(format=STAGE_FORMAT, stage=stage, policy=POLICY, **closure,
                       configuration_files=dict(asset_hashes, **_repository_references(root, closure["sources"])),
                       backend=_backend_projection(backend_identity, stage))
        payload["sha256"] = _digest(payload)
        result[stage] = payload
    result["manifest_sha256"] = _digest(result)
    return check_manifest(result)


def current_identity_manifest():
    from hecate_python_env import ROOT
    from poseidon_backend_identity import identity
    return identity_manifest(ROOT, identity())


def check_manifest(value):
    """Validate schema, complete stage/tool projections and every self hash."""
    _require(type(value) is dict and set(value) == {"format", "backend", "policy", *STAGES, "manifest_sha256"}
             and value["format"] == FORMAT and value["policy"] == POLICY, "Application identity format")
    _require(_sha(value["manifest_sha256"]) and value["manifest_sha256"] ==
             _digest({k: v for k, v in value.items() if k != "manifest_sha256"}), "Application identity hash")
    for stage in STAGES:
        item = value[stage]
        _require(type(item) is dict and set(item) == {"format", "stage", "policy", "roots", "sources",
            "import_edges", "dynamic_dependencies", "external_imports", "configuration_files", "backend", "sha256"},
            "Application stage fields")
        _require(item["format"] == STAGE_FORMAT and item["stage"] == stage and item["policy"] == POLICY,
                 "Application stage format")
        _require(_sha(item["sha256"]) and item["sha256"] ==
                 _digest({k: v for k, v in item.items() if k != "sha256"}), "Application stage hash")
        _require(item["backend"] == _backend_projection(value["backend"], stage), "Stage/backend projection")
        for key in ("sources", "configuration_files"):
            entries = item[key]
            _require(type(entries) is dict and all(_relative(p) and _sha(h) for p, h in entries.items()),
                     "Application source identity entries")
        _require(type(item["roots"]) is list and all(type(p) is str for p in item["roots"])
                 and item["roots"] == sorted(set(item["roots"]))
                 and item["roots"] and all(p in item["sources"] for p in item["roots"]), "Application stage roots")
        _require(type(item["import_edges"]) is dict and
                 all(p in item["sources"] and type(deps) is list and all(type(d) is str for d in deps) and deps == sorted(set(deps))
                     and all(d in item["sources"] for d in deps) for p, deps in item["import_edges"].items()),
                 "Application import closure")
        _require(type(item["dynamic_dependencies"]) is dict and
                 all(type(k) is str and type(v) is str for k, v in item["dynamic_dependencies"].items()),
                 "Application dynamic dependency declarations")
        _require(type(item["external_imports"]) is list and
                 all(type(k) is str for k in item["external_imports"]), "Application external imports")
    return value


def compatible(recorded, current, stages=DEFAULT_COMPATIBILITY):
    """Malformed identities raise; valid but different selected stages return False."""
    check_manifest(recorded)
    check_manifest(current)
    _require(type(stages) in (tuple, list) and 1 <= len(stages) <= len(STAGES)
             and all(type(s) is str and s in STAGES for s in stages) and len(set(stages)) == len(stages), "Identity comparison stages")
    return all(recorded[s]["sha256"] == current[s]["sha256"] for s in stages)
