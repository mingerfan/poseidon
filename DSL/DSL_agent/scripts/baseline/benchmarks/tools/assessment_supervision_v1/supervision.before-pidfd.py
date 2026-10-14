"""Linux process supervision for one isolated assessment command.

This module runs in a fresh child process. It observes and signals only that
process's descendants, including adopted children that created new sessions.
"""
import ctypes,hashlib,json,os,re,signal,stat,subprocess,sys,time
from pathlib import Path
FORMAT="poseidon-assessment-supervision-v1"
MAX_BYTES=65536
MAX_OWNED=1024
STAT_LIMIT=65536
CLEANUP_SECONDS=2.0
STATUS={"completed","interrupted","background_exit","cleanup_failed","supervision_error"}
class SupervisionError(RuntimeError):
    pass
class Interrupted(BaseException):
    pass
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()
def implementation():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def process_info(pid):
    try:
        fields=(Path("/proc")/str(pid)/"stat").read_text().rsplit(")",1)[1].split()
        return dict(pid=int(pid),parent=int(fields[1]),group=int(fields[2]),
                    start=int(fields[19]),state=fields[0])
    except (OSError,ValueError,IndexError):
        return None
def snapshot():
    result={}
    for index,path in enumerate(Path("/proc").iterdir()):
        if index>STAT_LIMIT:raise SupervisionError("process_inventory_limit")
        if path.name.isdigit():
            item=process_info(int(path.name))
            if item is not None:result[item["pid"]]=item
    return result
def same_process(pid,start):
    item=process_info(pid)
    return item is not None and item["start"]==start and item["state"]not in("Z","X")
def subreaper():
    # PR_SET_CHILD_SUBREAPER is scoped to this fresh Linux helper.
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:
        raise SupervisionError("subreaper_unavailable")
class OwnedDescendants:
    def __init__(self):
        self.pid=os.getpid();self.start=process_info(self.pid)["start"]
        self.owned={};self.detached=set();self.worker=None;self.worker_group=None
    def capture(self):
        rows=snapshot();owner=rows.get(self.pid)
        if owner is None or owner["start"]!=self.start:
            raise SupervisionError("supervisor_identity_changed")
        for pid,item in rows.items():
            current=pid;seen=set()
            while current not in seen and current!=self.pid and current in rows:
                seen.add(current);current=rows[current]["parent"]
            if current==self.pid and pid!=self.pid:
                self.owned[(pid,item["start"])]=True
                if self.worker is not None and pid!=self.worker and item["group"]!=self.worker_group:
                    self.detached.add((pid,item["start"]))
        if len(self.owned)>MAX_OWNED:raise SupervisionError("owned_process_limit")
    def living(self):
        return [(p,s)for p,s in self.owned if same_process(p,s)]
    def signal(self,kind):
        for pid,start in self.living():
            if same_process(pid,start):
                try:os.kill(pid,kind)
                except ProcessLookupError:pass
    def reap(self,worker_pid):
        for pid,start in self.owned:
            if pid==worker_pid:continue
            item=process_info(pid)
            if item is not None and item["start"]!=start:continue
            try:os.waitpid(pid,os.WNOHANG)
            except (ChildProcessError,ProcessLookupError):pass
    def finish(self,process):
        self.capture();initial=len(self.living());self.signal(signal.SIGTERM)
        deadline=time.monotonic()+CLEANUP_SECONDS
        gentle=min(deadline,time.monotonic()+.4)
        while time.monotonic()<gentle:
            self.capture();self.reap(process.pid)
            if not self.living():break
            time.sleep(.02)
        while time.monotonic()<deadline:
            self.capture();self.signal(signal.SIGKILL);self.reap(process.pid)
            if not self.living():break
            time.sleep(.02)
        process.wait(timeout=max(.1,deadline-time.monotonic()))
        self.capture();self.reap(process.pid)
        return initial,not self.living()
def receipt(token,invocation_binding,*,status,worker_exit_code,observed,detached,background,cleanup_completed):
    value=dict(format=FORMAT,token=token,invocation_binding=invocation_binding,
        implementation_sha256=implementation(),status=status,worker_exit_code=worker_exit_code,
        observed_processes=observed,detached_processes=detached,
        background_processes=background,cleanup_completed=cleanup_completed)
    value["binding"]=digest(value);return value
def checked(value,*,token,invocation_binding):
    fields={"format","token","invocation_binding","implementation_sha256","status","worker_exit_code",
            "observed_processes","detached_processes","background_processes","cleanup_completed","binding"}
    if type(value)is not dict or set(value)!=fields or value["format"]!=FORMAT or value["token"]!=token or value["invocation_binding"]!=invocation_binding or value["implementation_sha256"]!=implementation():
        raise SupervisionError("supervision_receipt_identity")
    if value["binding"]!=digest({k:v for k,v in value.items()if k!="binding"}) or value["status"]not in STATUS or type(value["cleanup_completed"])is not bool:
        raise SupervisionError("supervision_receipt_binding")
    for key in("observed_processes","detached_processes","background_processes"):
        if type(value[key])is not int or not 0<=value[key]<=MAX_OWNED:
            raise SupervisionError("supervision_receipt_count")
    exit_code=value["worker_exit_code"]
    if exit_code is not None and (type(exit_code)is not int or not -128<=exit_code<=255):
        raise SupervisionError("supervision_receipt_exit")
    if value["detached_processes"]>value["observed_processes"] or value["background_processes"]>value["observed_processes"]:
        raise SupervisionError("supervision_receipt_count")
    if value["status"]=="completed" and (not value["cleanup_completed"] or value["background_processes"] or exit_code is None):
        raise SupervisionError("supervision_receipt_completed")
    return value
def read_invocation(folder,token):
    if not re.fullmatch("[0-9a-f]{32}",token):raise SupervisionError("supervision_token")
    if folder!=folder.resolve() or folder.is_symlink() or not folder.is_dir():
        raise SupervisionError("supervision_directory")
    info=folder.stat()
    if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)&0o077:
        raise SupervisionError("supervision_directory")
    fd=os.open(folder/"invocation.json",os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,"rb")as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode)or info.st_size>MAX_BYTES:
            raise SupervisionError("supervision_invocation")
        raw=stream.read(MAX_BYTES+1)
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise SupervisionError("supervision_invocation")
            result[key]=value
        return result
    value=json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in()).throw(SupervisionError("supervision_invocation")))
    if value.get("token")!=token or value.get("binding")!=digest({k:v for k,v in value.items()if k!="binding"}):
        raise SupervisionError("supervision_invocation")
    return value
def write_receipt(folder,value):
    fd=os.open(folder/"process-supervision.json",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb")as stream:
        stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
def supervise(command,folder,token):
    invocation=read_invocation(folder,token)
    if not command or len(command)>64 or sum(len(x.encode())for x in command)>32768:
        raise SupervisionError("supervision_command")
    if Path(command[0]).resolve()!=Path(sys.executable).resolve():
        raise SupervisionError("supervision_executable")
    subreaper();owned=OwnedDescendants();process=None
    status="supervision_error";exit_code=None;background=0;clean=False
    def interrupted(signum,frame):raise Interrupted()
    for kind in(signal.SIGTERM,signal.SIGHUP,signal.SIGINT):
        signal.signal(kind,interrupted)
    try:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,start_new_session=True)
        owned.worker=process.pid
        worker=process_info(process.pid)
        owned.worker_group=None if worker is None else worker["group"]
        while process.poll()is None:
            owned.capture();time.sleep(.02)
        owned.capture();exit_code=process.returncode
        background=len(owned.living())
        status="background_exit"if background else"completed"
    except Interrupted:
        status="interrupted"
    except (OSError,ValueError,subprocess.SubprocessError,SupervisionError):
        status="supervision_error"
    finally:
        # Repeated caller signals cannot interrupt the bounded owned cleanup.
        for kind in(signal.SIGTERM,signal.SIGHUP,signal.SIGINT):
            signal.signal(kind,signal.SIG_IGN)
        if process is not None:
            try:
                _,clean=owned.finish(process)
                exit_code=process.returncode
            except (OSError,ValueError,subprocess.SubprocessError,SupervisionError):
                clean=False
        if not clean:status="cleanup_failed"
        value=receipt(token,invocation["binding"],status=status,worker_exit_code=exit_code,
            observed=len(owned.owned),detached=len(owned.detached),
            background=background,cleanup_completed=clean)
        write_receipt(folder,value)
    return exit_code if status=="completed"else 125
def main(argv=None):
    args=list(sys.argv[1:]if argv is None else argv)
    if len(args)<3:raise SupervisionError("supervision_arguments")
    return supervise(args[2:],Path(args[0]),args[1])
if __name__=="__main__":
    raise SystemExit(main())
