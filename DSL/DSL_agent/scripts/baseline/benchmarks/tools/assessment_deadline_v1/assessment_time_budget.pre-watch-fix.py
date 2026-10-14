"""Explicit persisted assessment deadlines and metadata-only CLI monitoring."""
import hashlib,json,math,os,re,stat,time
from pathlib import Path
FORMAT="poseidon-assessment-time-budget-v1"
PLAN_V1="poseidon-application-assessment-plan-v1"
PLAN_V2="poseidon-application-assessment-plan-v2"
MAX_SECONDS=14400
MAX_METADATA=16*1024**2
HASH=re.compile("[0-9a-f]{64}")
ID=re.compile("[0-9a-f]{32}")
BOOT=re.compile("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
TERMINAL={"succeeded","validation_failed","unsupported","cancelled","timed_out","service_error","budget_exhausted"}
class TimeBudgetError(ValueError):
    pass
def require(ok):
    if not ok:raise TimeBudgetError("assessment_time_budget_rejected")
def canonical(v):
    return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(v):
    return hashlib.sha256(canonical(v)).hexdigest()
def implementation():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def finite(v):
    return type(v) in (int,float) and math.isfinite(v)
def current_clock():
    try:
        boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        clock=time.clock_gettime(time.CLOCK_BOOTTIME);wall=time.time()
    except (AttributeError,OSError,ValueError):
        raise TimeBudgetError("assessment_clock_unavailable") from None
    require(BOOT.fullmatch(boot) is not None and finite(clock) and clock>=0 and finite(wall) and wall>0)
    return dict(boot=boot,clock=float(clock),wall=float(wall))
def create(seconds):
    require(type(seconds) is int and 1<=seconds<=MAX_SECONDS)
    now=current_clock()
    return dict(format=FORMAT,seconds=seconds,boot_id=now["boot"],
        created_wall=now["wall"],deadline_wall=now["wall"]+seconds,
        created_clock=now["clock"],deadline_clock=now["clock"]+seconds)
def checked(value):
    fields={"format","seconds","boot_id","created_wall","deadline_wall","created_clock","deadline_clock"}
    require(type(value) is dict and set(value)==fields and value["format"]==FORMAT)
    require(type(value["seconds"]) is int and 1<=value["seconds"]<=MAX_SECONDS and
        type(value["boot_id"]) is str and BOOT.fullmatch(value["boot_id"]) is not None)
    require(all(finite(value[n]) for n in ("created_wall","deadline_wall","created_clock","deadline_clock")) and
        value["created_wall"]>0 and value["created_clock"]>=0 and
        value["deadline_wall"]==value["created_wall"]+value["seconds"] and
        value["deadline_clock"]==value["created_clock"]+value["seconds"])
    return value
def plan_budget(plan):
    require(type(plan) is dict and plan.get("format") in (PLAN_V1,PLAN_V2))
    if plan["format"]==PLAN_V1:
        require("time_budget" not in plan)
        return None
    require("time_budget" in plan)
    return checked(plan["time_budget"])
def stop_reason(budget,*,cancelled=False):
    checked(budget)
    try:now=current_clock()
    except TimeBudgetError:return ("service_error","assessment_controller_clock_unavailable")
    if (now["boot"]!=budget["boot_id"] or now["clock"]<budget["created_clock"] or
            now["wall"]<budget["created_wall"]):
        return ("service_error","assessment_controller_clock_unavailable")
    if now["wall"]>=budget["deadline_wall"] or now["clock"]>=budget["deadline_clock"]:
        return ("timed_out","assessment_controller_deadline_reached")
    return ("cancelled","assessment_controller_cancelled") if cancelled else None
def _read(path):
    require(path==path.resolve() and not path.is_symlink())
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd);require(stat.S_ISREG(info.st_mode) and 0<info.st_size<=MAX_METADATA)
        with os.fdopen(fd,"rb",closefd=False) as f:raw=f.read(MAX_METADATA+1)
        require(len(raw)==info.st_size)
    finally:os.close(fd)
    def pairs(items):
        result={}
        for name,v in items:require(name not in result);result[name]=v
        return result
    try:value=json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(TimeBudgetError("assessment_time_budget_rejected")))
    except (TypeError,ValueError):raise TimeBudgetError("assessment_time_budget_rejected") from None
    require(type(value) is dict)
    return value
def _bound(v,expected=None):
    require(type(v.get("binding")) is str and HASH.fullmatch(v["binding"]) is not None and
        (expected is None or v["binding"]==expected) and
        v["binding"]==digest({k:x for k,x in v.items() if k!="binding"}))
    return v
class BudgetWatch:
    """Watch only a pinned plan, state and cancel marker; never load model inputs."""
    def __init__(self,store,evaluation_id,submission_binding):
        root=Path(store)
        require(root.is_absolute() and root==root.resolve() and not root.is_symlink() and
            type(evaluation_id) is str and ID.fullmatch(evaluation_id) is not None and
            type(submission_binding) is str and HASH.fullmatch(submission_binding) is not None)
        info=root.stat();require(stat.S_ISDIR(info.st_mode) and info.st_uid==os.getuid() and info.st_mode&0o077==0)
        require(_read(root/"store.json")==dict(format="poseidon-application-assessment-store-v1",path=str(root),uid=os.getuid()))
        self.path=root/evaluation_id;require(self.path==self.path.resolve() and not self.path.is_symlink())
        info=self.path.stat();require(stat.S_ISDIR(info.st_mode) and info.st_uid==os.getuid() and info.st_mode&0o077==0)
        self.plan=_bound(_read(self.path/"plan.json"),submission_binding)
        require(self.plan["evaluation_id"]==evaluation_id)
        self.budget=plan_budget(self.plan)
        self.evaluation_id=evaluation_id;self.binding=submission_binding
    def check(self):
        if self.budget is None:return None
        require(_bound(_read(self.path/"plan.json"),self.binding)==self.plan)
        state=_bound(_read(self.path/"state.json"))
        require(state.get("format")=="poseidon-application-assessment-state-v1" and
            state.get("submission_binding")==self.binding and
            state.get("status") in ({"queued","ready","running"}|TERMINAL))
        if state["status"] in TERMINAL:return None
        marker=self.path/"cancel.json";cancelled=marker.exists()
        if cancelled:
            require(_read(marker)==dict(format="poseidon-application-assessment-v1",evaluation_id=self.evaluation_id,
                submission_binding=self.binding,cancel_requested=True))
        return stop_reason(self.budget,cancelled=cancelled)
