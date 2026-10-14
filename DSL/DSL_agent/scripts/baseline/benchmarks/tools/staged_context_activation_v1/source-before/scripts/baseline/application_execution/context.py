"""Explicit pinned worker and request contexts; private host metadata only."""
import copy,hashlib,json,os,re,sys
from dataclasses import dataclass
from pathlib import Path

FORMAT="poseidon-worker-context-v1"
REQUEST_FORMAT="poseidon-request-context-v1"
MAX_BYTES=512*1024
ROOT=Path(__file__).resolve().parents[3]

class ContextError(RuntimeError):
    def __init__(self,code):
        self.code=code
        super().__init__(code)

def require(ok,code):
    if not ok:raise ContextError(code)

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def sha(raw):return hashlib.sha256(raw).hexdigest()

def _source_authority():
    from application_identity import source_closure,POLICY
    closure=source_closure(ROOT/"scripts/baseline",
        ("application_execution/context.py","application_execution/backend.py"))
    return dict(policy=POLICY,sources=closure["sources"],sha256=sha(canonical(closure)))

def _snapshot(store):
    from hecate_python_env import ROOT as runtime_root,WORK,VENV,PLATFORM
    from workspace_paths import WORK_BASE,work_root
    from platform_config import require_platform,identity,require_python_packages
    import numpy,torch
    from poseidon_runtime_paths import BUILD,CPU_BUILD,POSEIDON_BUILD,read_paths
    from application_backend import PoseidonBackend
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV and
            sys.version_info[:3]==(3,10,14),"context_worker_not_pinned")
    profile=require_platform()
    require_python_packages(torch,numpy)
    require(profile==PLATFORM and work_root()==WORK_BASE and
            WORK==WORK_BASE/profile["work_subdirectory"],"context_ambient_configuration_drift")
    require(runtime_root==ROOT,"context_source_checkout_mismatch")
    paths=read_paths()
    require(paths==dict(compiler_build=BUILD,executor_build=CPU_BUILD,poseidon_build=POSEIDON_BUILD),
            "context_tool_manifest_drift")
    require(type(store) in (str,Path) or isinstance(store,Path),"context_store_type")
    store=Path(store)
    require(store.is_absolute() and ".." not in store.parts and store==store.resolve(),
            "context_store_path")
    require(store!=Path(store.anchor) and store!=Path.home() and
            not store.is_relative_to(ROOT),"context_store_scope")
    require(not store.exists() or (store.is_dir() and not store.is_symlink()),"context_store_directory")
    # Only public configuration names are observed. Credential environment is never inspected.
    library={name:os.environ.get(name) for name in ("HECATE_PYTHON_LIBRARY_PATH","LD_LIBRARY_PATH")}
    require(type(library["HECATE_PYTHON_LIBRARY_PATH"]) is str and
            library["LD_LIBRARY_PATH"]==library["HECATE_PYTHON_LIBRARY_PATH"],"context_library_environment")
    backend=PoseidonBackend()
    stages=backend.identity_manifest()
    return dict(format=FORMAT,backend="poseidon_cpu",platform=identity(),
        paths=dict(source_root=str(ROOT),work_base=str(WORK_BASE),work_root=str(WORK),
            artifact_root=str(WORK/"results"),cache_root=str(WORK/"cache"),task_store=str(store),
            python_venv=str(VENV),**{k:str(v) for k,v in paths.items()}),
        library_environment=library,python_runtime=dict(python=list(sys.version_info[:3]),
            numpy=numpy.__version__,torch=torch.__version__,cuda=torch.version.cuda,
            numpy_module=str(Path(numpy.__file__).resolve()),torch_module=str(Path(torch.__file__).resolve())),
        backend_identity=backend.identity(),
        stage_identity_sha256=stages["manifest_sha256"],source_authority=_source_authority())

@dataclass(frozen=True)
class WorkerContext:
    """One configured SDK per worker; no runtime rebinding of module globals."""
    payload:bytes

    def __post_init__(self):
        require(type(self.payload) is bytes and len(self.payload)<=MAX_BYTES,"context_record_size")
        value=json.loads(self.payload)
        require(value.get("format")==FORMAT and canonical(value)==self.payload,"context_record_format")

    @classmethod
    def capture(cls,task_store):
        return cls(canonical(_snapshot(task_store)))

    @property
    def binding(self):return sha(self.payload)

    def record(self):
        return dict(**json.loads(self.payload),binding=self.binding)

    @property
    def task_store(self):return Path(json.loads(self.payload)["paths"]["task_store"])

    def check_current(self):
        require(canonical(_snapshot(self.task_store))==self.payload,"context_worker_changed")
        return self

    def request(self,request,level="numerical"):
        self.check_current()
        return RequestContext.capture(self,request,level)

    def persist(self):
        self.check_current()
        _write_checked(self.task_store/"worker-context.json",canonical(self.record()))
        return self.binding

@dataclass(frozen=True)
class RequestContext:
    payload:bytes

    def __post_init__(self):
        require(type(self.payload) is bytes and len(self.payload)<=8192,"request_context_record_size")
        value=json.loads(self.payload)
        require(value.get("format")==REQUEST_FORMAT and canonical(value)==self.payload,
                "request_context_record_format")

    @classmethod
    def capture(cls,worker,request,level):
        from component_contract import reconstruct_request
        require(isinstance(worker,WorkerContext),"request_context_worker_type")
        require(level in ("compiled","numerical"),"request_context_validation_level")
        require(type(request) is dict and request.get("compiler_configuration",{}).get("backend")=="poseidon_cpu",
                "request_context_backend")
        reconstruct_request(copy.deepcopy(request))
        return cls(canonical(dict(format=REQUEST_FORMAT,worker_binding=worker.binding,
            request_id=request["request_id"],request_sha256=sha(canonical(request)),
            compiler_configuration_sha256=sha(canonical(request["compiler_configuration"])),
            layout_sha256=sha(canonical(request["layout"])),validation_level=level)))

    @property
    def binding(self):return sha(self.payload)

    def record(self):return dict(**json.loads(self.payload),binding=self.binding)

    def check(self,worker,request,level):
        worker.check_current()
        require(self.payload==RequestContext.capture(worker,request,level).payload,"request_context_changed")
        return self

def check_request_record(value,worker,request,level):
    require(type(value) is dict and type(value.get("binding")) is str,"request_context_receipt_type")
    raw=canonical({k:v for k,v in value.items() if k!="binding"})
    require(sha(raw)==value["binding"],"request_context_receipt_hash")
    context=RequestContext(raw)
    return context.check(worker,request,level)

def _write_checked(path,raw):
    """Create a private receipt once; matching receipts can be reused."""
    from model_weight_package import regular_bytes
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    require(path.parent==path.parent.resolve() and not path.parent.is_symlink(),"context_receipt_directory")
    if path.exists() or path.is_symlink():
        require(regular_bytes(path,MAX_BYTES)==raw,"context_receipt_conflict")
        return
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,"wb") as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())

def request_receipt(worker,value):
    require(re.fullmatch("[0-9a-f]{64}",value["binding"]) is not None,"request_context_receipt_name")
    path=worker.task_store/"request-contexts"/(value["binding"]+".json")
    _write_checked(path,canonical(value))
    return path
