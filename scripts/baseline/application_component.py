"""Persistent single-worker application boundary over the existing synthesis loop.

Host-owned provider/backend objects only; no credentials, plug-in paths or live
CLI action. Separate OS processes are required for execution and cancellation.
"""
import contextlib,copy,fcntl,hashlib,json,os,re,signal,time,uuid,traceback
from pathlib import Path
from benchmark_graph import canonical,digest,require
from candidate_contract import strict_json
from component_contract import synthesize,validate_program
from application_policy import budget,prepare,structured_failure,decision,BACKEND
TERMINAL={"succeeded","unsupported","budget_exhausted","validation_failed","service_error","cancelled","timed_out"}
class Stopped(BaseException):
    def __init__(self,status):self.status=status

class Application:
    def __init__(self,store,backend):
        self.store=Path(store).resolve();self.store.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.backend=backend
    def _dir(self,task):
        require(type(task) is str and re.fullmatch("[0-9a-f]{32}",task),"Task identifier")
        p=self.store/task
        require(not p.is_symlink(),"Indirect task directory")
        return p
    def _write(self,path,value):
        raw=canonical(value)
        tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
        with tmp.open("xb") as f:
            os.chmod(tmp,0o600);f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    def _load(self,path):
        require(path.is_file() and not path.is_symlink() and path.stat().st_size<=4*1024**2,"Task file")
        return strict_json(path.read_text())
    @contextlib.contextmanager
    def _lock(self,p):
        with (p/"lock").open("a") as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)
    def submit(self,model,*,backend=BACKEND,validation_level="numerical",options=None,limits=None):
        task=uuid.uuid4().hex;p=self._dir(task);p.mkdir(mode=0o700)
        state=dict(task_id=task,status="queued",phase="input_check",validation_level=validation_level,
                   generations=0,http_attempts=0,ledger=[],failure=None,package=None,acceptance=None)
        self._write(p/"state.json",state)
        try:
            b=budget(limits if limits is not None else dict(generations=4,http_attempts=16,wall_seconds=3600,
                               api_timeout_seconds=1200,max_tokens=384000))
            q,context=prepare(model,backend,validation_level,options)
        except (ValueError,TypeError,KeyError,IndexError):
            state.update(status="unsupported",phase="complete",
                         failure=dict(code="input_or_capability_unsupported",layer="input",location=None,context={}))
            self._write(p/"state.json",state);return task
        try:identity=self.backend.identity()
        except Exception:
            state.update(status="service_error",phase="complete",
                         failure=dict(code="backend_unavailable",layer="environment",location=None,context={}))
            self._write(p/"state.json",state);return task
        payload=dict(request=q,context=context,budget=b,identity=identity,validation_level=validation_level)
        payload["binding"]=digest(payload);self._write(p/"task.json",payload)
        state["reuse_key"]=digest(dict(request_id=q["request_id"],identity=identity,validation_level=validation_level))
        state["phase"]="queued";self._write(p/"state.json",state)
        return task
    def status(self,task):
        s=self._load(self._dir(task)/"state.json")
        return {k:copy.deepcopy(s[k]) for k in ("task_id","status","phase","validation_level",
                       "generations","http_attempts","failure","package","acceptance")}
    def result(self,task):
        s=self.status(task)
        if s["status"]=="succeeded":
            try:
                p=self._dir(s["package"]["task_id"]);meta=self._load(p/"task.json")
                require(self.backend.identity()==meta["identity"],"Dependency changed")
                require(self.backend.check_package(p/"program",s["validation_level"])==s["package"]["binding"],"Package changed")
            except Exception:
                s.update(status="service_error",package=None,failure=dict(code="package_identity_mismatch",layer="delivery"))
        return s
    def cancel(self,task):
        p=self._dir(task)
        self._write(p/"cancel.json",dict(requested=True))
        try:
            with self._lock(p):
                s=self._load(p/"state.json")
                if s["status"]=="queued":
                    s.update(status="cancelled",phase="complete");self._write(p/"state.json",s)
        except BlockingIOError:pass
        return self.status(task)
    def recover(self,task):
        # A lost worker can have a billed HTTP request. Never replay it automatically.
        p=self._dir(task)
        with self._lock(p):
            s=self._load(p/"state.json")
            if s["status"]=="running":
                s.update(status="service_error",phase="complete",
                         failure=dict(code="interrupted_outcome_unknown",layer="worker"))
                self._write(p/"state.json",s)
        return self.status(task)
    def run(self,task,provider):
        p=self._dir(task)
        with self._lock(p):
            state=self._load(p/"state.json")
            if state["status"] in TERMINAL:return self.result(task)
            require(state["status"]=="queued","Recover interrupted worker explicitly")
            try:
                meta=self._load(p/"task.json")
                require(digest({k:v for k,v in meta.items() if k!="binding"})==meta["binding"],"Task binding")
                b=budget(meta["budget"]);level=meta["validation_level"]
            except (ValueError,TypeError,KeyError,OSError):
                state.update(status="service_error",phase="complete",
                             failure=dict(code="task_integrity_rejected",layer="integrity"))
                self._write(p/"state.json",state);return self.status(task)
            state.update(status="running",phase="preflight");self._write(p/"state.json",state)
            control=Controller(self,p,state,b)
            previous=None;alarm=None;old_before=None;old_after=None
            try:
                control.check()
                require(self.backend.identity()==meta["identity"],"Backend changed")
                # Reuse is explicit in the result and always rechecks package integrity.
                for other in sorted(self.store.glob("*/state.json")):
                    control.check()
                    if other.parent==p:continue
                    prior=self._load(other)
                    if prior.get("reuse_key")==state["reuse_key"] and prior["status"]=="succeeded":
                        origin=prior["package"]["task_id"];package=self._dir(origin)/"program"
                        binding=self.backend.check_package(package,level)
                        require(binding==prior["package"]["binding"],"Cached package changed")
                        control.check()
                        state.update(status="succeeded",phase="complete",failure=None,acceptance=prior["acceptance"],
                                     package=dict(task_id=origin,binding=binding,reused=True))
                        self._write(p/"state.json",state);return self.result(task)
                require(getattr(provider,"kind",None) in ("scripted_replay","deepseek_api","deepseek_offline"),
                        "Only the configured provider or explicit offline replay is supported")
                # Cancellation/deadline while blocked in a provider or backend operation.
                require(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),"Worker already owns an alarm")
                def alarm_check(*_):
                    try:control.check()
                    except Stopped:
                        signal.setitimer(signal.ITIMER_REAL,0)
                        raise
                previous=signal.signal(signal.SIGALRM,alarm_check)
                alarm=signal.setitimer(signal.ITIMER_REAL,.1,.1)
                if getattr(provider,"kind",None)=="deepseek_api":
                    from deepseek_provider import DeepSeekProvider
                    require(isinstance(provider,DeepSeekProvider),"Supported paid provider adapter required")
                    c=provider.config
                    require(provider.generation_attempts==provider.request_attempts==0,"Fresh provider session")
                    require(c.max_calls<=b["generations"] and c.max_tokens<=b["max_tokens"] and
                            c.timeout_seconds<=b["api_timeout_seconds"],"Provider exceeds task budget")
                if hasattr(provider,"before_attempt"):
                    old_before=provider.before_attempt;old_after=provider.on_attempt
                    require(old_before is None and old_after is None,"Provider hooks already owned")
                    provider.before_attempt=control.reserve_http
                    provider.on_attempt=lambda:control.provider_checkpoint(provider.metrics())
                last={}
                def evaluate(raw,index):
                    control.check();state["phase"]="validating";control.save()
                    try:
                        candidate=strict_json(raw)
                        static=validate_program(candidate,meta["request"])
                    except (ValueError,TypeError,KeyError,IndexError):
                        result=dict(status="failed",failure=dict(layer="response_parse"),
                            public_feedback=dict(status="failed",layer="response_parse",category="candidate",
                              diagnostic="Candidate response does not match the fixed response contract"))
                    else:
                        if static["status"]=="rejected":
                            result=dict(status="failed",failure=static["failure"],
                                public_feedback=dict(status="failed",layer="static_check",category="candidate",
                                  diagnostic="Candidate violates the fixed DSL contract"))
                        else:
                            # Backend exceptions belong to the host, not generated syntax.
                            result=self.backend.qualify(meta["request"],candidate,level)
                    last.clear();last.update(result)
                    control.last_failure=structured_failure(result)
                    # Preserve full evidence locally; status() never returns it.
                    self._write(p/("validation-%d.json"%index),result)
                    return result
                def record(index,raw,feedback):
                    self._write(p/("response-%d.json"%index),dict(raw=raw,feedback=feedback))
                    control.save()
                loop=synthesize(meta["request"],provider,evaluate,record,max_repairs=b["generations"]-1,
                                validation_level=level,control=control)
                control.check()
                if loop["status"]=="passed":
                    state["phase"]="packaging";control.save()
                    self.backend.export(last,p/"program",level);control.check()
                    binding=self.backend.check_package(p/"program",level)
                    require(self.backend.identity()==meta["identity"],"Dependency changed during task")
                    control.check()
                    acceptance=dict(level=level,compiled=True,artifact_checked=True,
                        encrypted_execution=level=="numerical",numerically_validated=level=="numerical",
                        input_groups=4 if level=="numerical" else 0,
                        compared_values=last.get("compared_values",0),
                        maximum_absolute_error=last.get("max_absolute_error"),
                        claim=("finite configured inputs only; no formal equivalence" if level=="numerical"
                               else "compilation and artifacts only; no numerical claim"))
                    state.update(status="succeeded",failure=None,acceptance=acceptance,
                                 package=dict(task_id=task,binding=binding,reused=False))
                else:
                    status={"repair_budget_exhausted":"budget_exhausted","provider_exhausted":"service_error",
                            "provider_failed":"service_error","integrity_failed":"service_error",
                            "infrastructure_failed":"service_error"}.get(loop["status"],loop["status"])
                    state["status"]=status if status in TERMINAL else "service_error"
                    state["failure"]=control.last_failure or dict(code="provider_or_worker_failure",layer="provider")
            except Stopped as stop:state["status"]=stop.status
            except Exception as error:
                self._write(p/"host-error.json",dict(error_type=type(error).__name__,
                    frames=[dict(file=Path(f.filename).name,line=f.lineno) for f in traceback.extract_tb(error.__traceback__)]))
                state.update(status="service_error",failure=dict(code="host_operation_failed",layer="environment"))
            finally:
                if previous is not None:
                    signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
                if hasattr(provider,"before_attempt"):
                    provider.before_attempt=old_before;provider.on_attempt=old_after
                state["phase"]="complete";control.save()
        return self.result(task)

class Controller:
    def __init__(self,app,path,state,bounds):
        self.app=app;self.path=path;self.state=state;self.bounds=bounds
        self.deadline=time.monotonic()+bounds["wall_seconds"]
        self.seen=set();self.last_failure=None;self.last_key=None;self.repeated=0
    def observe_failure(self,result):self.last_failure=structured_failure(result)
    def save(self):self.app._write(self.path/"state.json",self.state)
    def check(self):
        if (self.path/"cancel.json").exists():raise Stopped("cancelled")
        if time.monotonic()>=self.deadline:raise Stopped("timed_out")
    def before_generation(self,index):
        self.check()
        if self.state["generations"]>=self.bounds["generations"]:return "budget_exhausted"
        self.state["generations"]+=1;self.state["phase"]="generating"
        self.state["ledger"].append(dict(event="generation_reserved",index=index));self.save()
    def reserve_http(self):
        self.check()
        if self.state["http_attempts"]>=self.bounds["http_attempts"]:raise Stopped("budget_exhausted")
        self.state["http_attempts"]+=1
        self.state["ledger"].append(dict(event="http_reserved",index=self.state["http_attempts"]-1,
                                        outcome="unknown_until_checkpoint"));self.save()
    def provider_checkpoint(self,metrics):
        # Fixed count/status fields only. Full raw transport diagnostics stay with provider.
        calls=metrics.get("calls",[])
        self.state["ledger"].append(dict(event="provider_checkpoint",attempts=metrics.get("request_attempts",0),
             outcomes=[{k:c[k] for k in ("index","status","error") if k in c} for c in calls]))
        self.save();self.check()
    def before_evaluation(self,raw,index):
        self.check()
        if type(raw) is str:
            record=dict(raw=raw) if len(raw.encode())<=262144 else dict(
                oversized_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),bytes=len(raw.encode()))
            self.app._write(self.path/("generation-%d.json"%index),record)
        try:
            obj=strict_json(raw);fingerprint=hashlib.sha256(obj["hecate_source"].encode()).hexdigest()
        except (ValueError,TypeError,KeyError,AttributeError):
            fingerprint=hashlib.sha256(str(raw).encode()).hexdigest()
        if fingerprint in self.seen:
            self.last_failure=dict(code="repeated_candidate",layer="generation",retry_policy="stop")
            return "validation_failed"
        self.seen.add(fingerprint)
    def after_evaluation(self,feedback,index):
        self.check()
        f=structured_failure(dict(failure=self.last_failure or {},public_feedback=feedback))
        # Preserve trusted structured failure when available; never use its arbitrary diagnostic.
        if self.last_failure and self.last_failure.get("retry_policy"):f=structured_failure(dict(failure=self.last_failure))
        self.last_failure=f;self.state["failure"]=f
        route=decision(f)
        if route!="repair":return route
        key=digest({k:f.get(k) for k in ("layer","code","location","context")})
        self.repeated=self.repeated+1 if key==self.last_key else 1;self.last_key=key
        if self.repeated>=2:
            self.last_failure=dict(f,stop_reason="repeated_failure")
            self.state["failure"]=self.last_failure
            return "validation_failed"
        self.state["phase"]="repairing";self.save()
