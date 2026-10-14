"""Persistent single-worker application boundary over the existing synthesis loop.

Host-owned provider/backend objects only; no credentials, plug-in paths or live
CLI action. Separate OS processes are required for execution and cancellation.
"""
import contextlib,copy,fcntl,hashlib,json,math,os,re,signal,time,uuid,traceback
from pathlib import Path
from benchmark_graph import canonical,digest,require
from candidate_contract import strict_json
from component_contract import synthesize,validate_program
from application_policy import budget,prepare,structured_failure,decision,BACKEND
from application_diagnostics import LEGACY, PRECISE, POLICIES, attach, repeat_key
from execution_input_domain import ApproximationDomainRejected
from application_admission import (LEGACY as ADMISSION_LEGACY, STRUCTURED as ADMISSION_STRUCTURED,
                                   AdmissionRejected, diagnostic as admission_diagnostic,
                                   policy as check_admission_policy, precheck as admission_precheck)
from application_artifact_budget import (ArtifactMonitor, OperationalStopped,
                                        limits as operational_budget)
TERMINAL={"succeeded","unsupported","budget_exhausted","validation_failed","service_error","cancelled","timed_out"}
class Stopped(BaseException):
    def __init__(self,status):self.status=status

class PackageBindingRejected(ValueError):
    def __init__(self,code):
        require(code in {"cached_task_binding_rejected","cached_request_mismatch",
                         "cached_provenance_mismatch","cached_level_mismatch","cached_dependency_mismatch"},
                "Package binding rejection code")
        self.code=code
        super().__init__(code)
    def public_failure(self):
        return dict(code=self.code,layer="delivery",owner="host",retry_policy="never_automatic",
                    location=None,context={})

from application_model_source import ModelSourceIntake

class Application(ModelSourceIntake):
    def __init__(self,store,backend):
        self.store=Path(store).resolve();self.store.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.backend=backend
    def _dir(self,task):
        require(type(task) is str and re.fullmatch("[0-9a-f]{32}",task),"Task identifier")
        p=self.store/task
        require(not p.is_symlink(),"Indirect task directory")
        return p
    def _write(self,path,value):
        # Ordered source provenance is data with a declaration sequence.
        # Preserve it in task transport; canonical request/binding hashes and
        # all status, budget and ledger serialization retain their old policy.
        if path.name == "task.json" and type(value) is dict and "model_provenance" in value:
            raw=json.dumps(value,sort_keys=False,separators=(",",":"),allow_nan=False).encode()
        else:
            raw=canonical(value)
        tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
        with tmp.open("xb") as f:
            os.chmod(tmp,0o600);f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    def _load(self,path):
        require(path.is_file() and not path.is_symlink() and path.stat().st_size<=4*1024**2,"Task file")
        def pairs(items):
            value={}
            for key,item in items:
                require(key not in value, "Duplicate task JSON key")
                value[key]=item
            return value
        def nonfinite(_):
            raise ValueError("Nonfinite task JSON")
        def real(token):
            value=float(token)
            require(math.isfinite(value), "Nonfinite task JSON")
            return value
        try:
            return json.loads(path.read_text(),object_pairs_hook=pairs,
                              parse_constant=nonfinite,parse_float=real)
        except (RecursionError,MemoryError) as error:
            raise ValueError("Task JSON resource limit") from error
    @contextlib.contextmanager
    def _lock(self,p):
        with (p/"lock").open("a") as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)
    def _identity(self, policy):
        if policy == "staged-v1":
            from application_identity import check_manifest
            return check_manifest(self.backend.identity_manifest())
        require(policy == "legacy-v1", "Identity policy")
        return self.backend.identity()
    def _checked_task_metadata(self,task):
        meta=self._load(self._dir(task)/"task.json")
        require(type(meta) is dict and digest({k:v for k,v in meta.items() if k!="binding"})==meta.get("binding"),
                "Task binding")
        if "model_source" in meta:
            from model_source_provenance import build as source_origin
            snapshot=self._load(self._dir(task)/"source.json")
            require(type(snapshot) is dict and
                    digest({k:v for k,v in snapshot.items() if k!="binding"})==snapshot.get("binding"),
                    "Source snapshot binding")
            require(source_origin(snapshot,meta["request"],meta.get("model_provenance"))==meta["model_source"],
                    "Task/source origin mismatch")
        return meta
    def _package_task_metadata(self,origin,requested,level):
        """Verify cache/delivery identity against the immutable originating request."""
        try:meta=self._checked_task_metadata(origin)
        except (ValueError,TypeError,KeyError,OSError) as error:
            raise PackageBindingRejected("cached_task_binding_rejected") from error
        if meta.get("validation_level")!=level or requested.get("validation_level")!=level:
            raise PackageBindingRejected("cached_level_mismatch")
        if (type(meta.get("request")) is not dict or type(requested.get("request")) is not dict or
                digest(meta["request"])!=digest(requested["request"])):
            raise PackageBindingRejected("cached_request_mismatch")
        if digest(meta.get("model_provenance"))!=digest(requested.get("model_provenance")):
            raise PackageBindingRejected("cached_provenance_mismatch")
        from model_source_provenance import reuse_identity
        def source_key(value):
            return None if value is None else reuse_identity(value)
        if digest(source_key(meta.get("model_source")))!=digest(source_key(requested.get("model_source"))):
            raise PackageBindingRejected("cached_provenance_mismatch")
        if not self._identity_matches(meta,delivery=True):
            raise PackageBindingRejected("cached_dependency_mismatch")
        return meta
    def _identity_matches(self, meta, *, delivery=False):
        source=meta.get("model_source")
        if source is not None:
            from application_model_source import own_identity
            from model_source_provenance import verify
            try:
                verify(source,meta["request"],meta.get("model_provenance"),
                       controller_sha256=own_identity()["controller_sha256"])
            except (ValueError,TypeError,KeyError,OSError):
                return False
        policy=meta.get("identity_policy", "legacy-v1")
        actual=self._identity(policy)
        if delivery and policy == "staged-v1":
            from application_identity import compatible
            return compatible(meta["identity"], actual)
        return actual == meta["identity"]
    def submit(self,model,*,backend=BACKEND,validation_level="numerical",options=None,limits=None,
               feedback_policy=PRECISE,model_provenance=None,admission_policy=ADMISSION_LEGACY,
               operational_limits=None):
        task=uuid.uuid4().hex;p=self._dir(task);p.mkdir(mode=0o700)
        state=dict(task_id=task,status="queued",phase="input_check",validation_level=validation_level,
                   generations=0,http_attempts=0,ledger=[],failure=None,package=None,acceptance=None,generation=None)
        self._write(p/"state.json",state)
        try:
            check_admission_policy(admission_policy)
            requested_limits=limits if limits is not None else dict(generations=4,http_attempts=16,wall_seconds=3600,
                               api_timeout_seconds=1200,max_tokens=384000)
            if admission_policy == ADMISSION_STRUCTURED:
                admission_precheck(model,backend=backend,validation_level=validation_level,limits=requested_limits)
            if operational_limits is not None:
                try:operational_limits=operational_budget(operational_limits)
                except (ValueError,TypeError,KeyError):
                    raise AdmissionRejected("operational_limits",location="operational_limits") from None
            require(feedback_policy in POLICIES,"Feedback policy")
            b=budget(requested_limits)
            q,context=prepare(model,backend,validation_level,options)
            if model_provenance is not None:
                from candidate_bundle import check_provenance
                from public_weight_context import storage_limit
                require(len(canonical(model_provenance))<=storage_limit(q),"Model provenance budget")
                model_provenance=copy.deepcopy(model_provenance)
                check_provenance(model_provenance,q["model"])
        except AdmissionRejected as error:
            state.update(status="unsupported",phase="complete",failure=error.public_failure)
            self._write(p/"state.json",state);return task
        except (ValueError,TypeError,KeyError,IndexError):
            fallback=(admission_diagnostic("unknown") if admission_policy == ADMISSION_STRUCTURED else
                      dict(code="input_or_capability_unsupported",layer="input",location=None,context={}))
            state.update(status="unsupported",phase="complete",failure=fallback)
            self._write(p/"state.json",state);return task
        identity_policy = "staged-v1" if callable(getattr(self.backend,"identity_manifest",None)) else "legacy-v1"
        try:identity=self._identity(identity_policy)
        except Exception:
            state.update(status="service_error",phase="complete",
                         failure=dict(code="backend_unavailable",layer="environment",location=None,context={}))
            self._write(p/"state.json",state);return task
        payload=dict(request=q,context=context,budget=b,identity=identity,identity_policy=identity_policy,validation_level=validation_level,
                     feedback_policy=feedback_policy)
        key=dict(request_id=q["request_id"],identity=identity,validation_level=validation_level)
        if model_provenance is not None:
            payload["model_provenance"]=model_provenance
            key["model_provenance_sha256"]=digest(model_provenance)
        if admission_policy != ADMISSION_LEGACY:
            payload["admission_policy"]=admission_policy
        if operational_limits is not None:
            payload["operational_limits"]=operational_limits
        payload["binding"]=digest(payload)
        if len(canonical(payload))>4*1024**2:
            state.update(status="unsupported",phase="complete",
                         failure=dict(code="task_storage_capacity_exceeded",layer="input",location=None,context={}))
            self._write(p/"state.json",state);return task
        self._write(p/"task.json",payload)
        state["reuse_key"]=digest(key)
        state["phase"]="queued"
        if operational_limits is not None:
            try:
                monitor=ArtifactMonitor(self,p,operational_limits,initialize=True)
                monitor.check(force=True)
                state["operational_usage"]=monitor.usage()
            except OperationalStopped as stop:
                state.update(status=stop.status,phase="complete",failure=stop.public_failure)
        self._write(p/"state.json",state)
        return task
    def submit_import(self,model_import,**requirements):
        """Submit a checked offline import; retained Python sources stay inert."""
        try:
            from candidate_bundle import load_model_provenance
            from strict_json_files import strict_file
            folder=Path(model_import)
            from public_weight_context import POLICIES as PUBLIC_WEIGHT_POLICIES,TRUSTED_BYTES
            options=requirements.get("options") or {}
            max_bytes=TRUSTED_BYTES if type(options) is dict and options.get("public_weight_policy") in PUBLIC_WEIGHT_POLICIES else 131072
            model=strict_file(folder/"model.json",max_bytes)
            provenance=load_model_provenance(folder,model,max_bytes=max_bytes)
            from pytorch_model_intake import preparation_options
            requirements["options"]=preparation_options(provenance,model,options)
            from operator_arithmetic_graph import preparation_options as graph_options
            requirements["options"]=graph_options(provenance["original"],model,requirements["options"])
        except (ValueError,TypeError,KeyError,IndexError,OSError):
            return self.submit(None,**requirements)
        return self.submit(model,model_provenance=provenance,**requirements)

    def status(self,task):
        s=self._load(self._dir(task)/"state.json")
        result={k:copy.deepcopy(s[k]) for k in ("task_id","status","phase","validation_level",
                       "generations","http_attempts","failure","package","acceptance")} | {
                           "generation":copy.deepcopy(s.get("generation"))}
        if "intake" in s:result["intake"]=copy.deepcopy(s["intake"])
        if "operational_usage" in s:result["operational_usage"]=copy.deepcopy(s["operational_usage"])
        return result
    def result(self,task):
        s=self.status(task)
        if s["status"]=="succeeded":
            try:
                requested=self._checked_task_metadata(task)
                p=self._dir(s["package"]["task_id"])
                self._package_task_metadata(s["package"]["task_id"],requested,s["validation_level"])
                require(self._identity_matches(requested, delivery=True),"Dependency changed")
                require(self.backend.check_package(p/"program",s["validation_level"])==s["package"]["binding"],"Package changed")
            except PackageBindingRejected as error:
                s.update(status="service_error",package=None,failure=error.public_failure())
            except Exception:
                s.update(status="service_error",package=None,failure=dict(code="package_identity_mismatch",layer="delivery"))
        return s
    def execute(self,task,samples,*,operational_limits=None,sample_format="named"):
        """Package execution; an explicit resource budget applies to this call."""
        if operational_limits is None:
            if sample_format=="named":return self._execute(task,samples)
            return self._execute(task,samples,sample_format=sample_format)
        try:
            bounds=operational_budget(operational_limits)
        except (ValueError,TypeError,KeyError):
            return dict(status="unsupported",failure=admission_diagnostic("operational_limits",
                        location="operational_limits"),agent_calls=0,compiler_calls=0,encrypted_execution=False)
        try:
            meta=self._checked_task_metadata(task)
        except (ValueError,TypeError,KeyError,OSError):
            return dict(status="service_error",failure=dict(code="task_integrity_rejected",layer="integrity"),
                        agent_calls=0,compiler_calls=0,encrypted_execution=False)
        folder=self._dir(task)/("execution-control-"+uuid.uuid4().hex)
        folder.mkdir(mode=0o700)
        monitor=None;previous=None;result=None
        deadline=time.monotonic()+budget(meta["budget"])["wall_seconds"]
        def check(*_):
            if (self._dir(task)/"cancel.json").exists():raise Stopped("cancelled")
            if time.monotonic()>=deadline:raise Stopped("timed_out")
            if monitor is not None:monitor.check()
        def alarm_check(*_):
            try:check()
            except (Stopped,OperationalStopped):
                signal.setitimer(signal.ITIMER_REAL,0)
                raise
        def observe(path):
            check()
            monitor.register(path)
        try:
            monitor=ArtifactMonitor(self,folder,bounds,initialize=True)
            monitor.check(force=True)
            require(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),"Worker already owns an alarm")
            previous=signal.signal(signal.SIGALRM,alarm_check)
            signal.setitimer(signal.ITIMER_REAL,.1,.1)
            check()
            execute_options={}if sample_format=="named"else dict(sample_format=sample_format)
            result=self._execute(task,samples,artifact_observer=observe,**execute_options)
            monitor.check(force=True)
            result["operational_usage"]=monitor.usage()
        except OperationalStopped as stop:
            result=dict(status=stop.status,failure=stop.public_failure,
                        agent_calls=0,compiler_calls=0,
                        encrypted_execution_observed=(result.get("encrypted_execution") if result else None))
        except Stopped as stop:
            result=dict(status=stop.status,agent_calls=0,compiler_calls=0,
                        encrypted_execution_observed=(result.get("encrypted_execution") if result else None))
        except Exception:
            result=dict(status="service_error",failure=dict(code="program_execution_unavailable",layer="environment"),
                        agent_calls=0,compiler_calls=0)
        finally:
            if previous is not None:
                signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
            if result is not None:
                if monitor is not None:result["operational_usage"]=monitor.usage()
                self._write(folder/"state.json",{k:copy.deepcopy(result[k]) for k in
                    ("status","failure","operational_usage","agent_calls","compiler_calls","encrypted_execution_observed")
                    if k in result})
        return result

    def _execute(self,task,samples,*,artifact_observer=None,sample_format="named"):
        """Local pinned-worker inference using a completed, checked package."""
        if sample_format not in("named","forward_call"):
            return dict(status="unsupported",failure=dict(code="input_format_unsupported",layer="input"),
                        agent_calls=0,compiler_calls=0,encrypted_execution=False)
        result=self.result(task)
        if result["status"] != "succeeded":
            return dict(status="validation_failed", failure=dict(code="program_not_available",layer="delivery"))
        from application_planning.compiled_resources import CompiledStorageRejected, CompiledResourceIntegrityRejected
        try:
            package=self._dir(result["package"]["task_id"])/"program"
            options={} if artifact_observer is None else dict(artifact_observer=artifact_observer)
            if sample_format!="named":options["sample_format"]=sample_format
            return self.backend.execute(package,samples,**options)
        except CompiledResourceIntegrityRejected:
            from application_diagnostics import make
            return dict(status="service_error", failure=make("integrity_rejected"),
                        encrypted_execution=False,agent_calls=0,compiler_calls=0)
        except CompiledStorageRejected as exc:
            from application_diagnostics import checked
            try:
                diagnostic=checked(exc.structured_diagnostic())
                require(diagnostic["reason_code"]=="compiled_storage_budget", "Storage rejection diagnostic")
            except (ValueError,TypeError,KeyError,IndexError):
                return dict(status="service_error",
                            failure=dict(code="program_execution_unavailable",layer="environment"))
            return dict(status="validation_failed", failure=diagnostic,
                        encrypted_execution=False,agent_calls=0,compiler_calls=0)
        except ApproximationDomainRejected as exc:
            return dict(status="validation_failed", failure=exc.public_failure,
                        encrypted_execution=False,agent_calls=0,compiler_calls=0)
        except (ValueError,TypeError,KeyError,IndexError):
            return dict(status="validation_failed",
                        failure=dict(code="program_input_or_execution_rejected",layer="execution"))
        except Exception:
            return dict(status="service_error",
                        failure=dict(code="program_execution_unavailable",layer="environment"))
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
        if (p/"source.json").exists():
            self.prepare_source(task)
            if self.status(task)["status"] in TERMINAL:return self.result(task)
        with self._lock(p):
            state=self._load(p/"state.json")
            if state["status"] in TERMINAL:return self.result(task)
            require(state["status"]=="queued","Recover interrupted worker explicitly")
            try:
                meta=self._checked_task_metadata(task)
                b=budget(meta["budget"]);level=meta["validation_level"]
                policy=meta.get("feedback_policy",LEGACY)
                require(policy in POLICIES,"Feedback policy")
                if "admission_policy" in meta:check_admission_policy(meta["admission_policy"])
                if "operational_limits" in meta:operational_budget(meta["operational_limits"])
            except (ValueError,TypeError,KeyError,OSError):
                state.update(status="service_error",phase="complete",
                             failure=dict(code="task_integrity_rejected",layer="integrity"))
                self._write(p/"state.json",state);return self.status(task)
            state.update(status="running",phase="preflight");self._write(p/"state.json",state)
            try:
                control=Controller(self,p,state,b,policy=policy,operational_limits=meta.get("operational_limits"))
            except OperationalStopped as stop:
                state.update(status=stop.status,phase="complete",failure=stop.public_failure)
                self._write(p/"state.json",state);return self.status(task)
            previous=None;alarm=None;old_before=None;old_after=None
            try:
                # Cover preflight and cached delivery as well as generation/validation.
                require(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),"Worker already owns an alarm")
                def alarm_check(*_):
                    try:control.check()
                    except (Stopped,OperationalStopped):
                        signal.setitimer(signal.ITIMER_REAL,0)
                        raise
                previous=signal.signal(signal.SIGALRM,alarm_check)
                alarm=signal.setitimer(signal.ITIMER_REAL,.1,.1)
                control.check()
                require(self._identity_matches(meta),"Backend changed")
                # Reuse is explicit in the result and always rechecks package integrity.
                for other in sorted(self.store.glob("*/state.json")):
                    control.check()
                    if other.parent==p:continue
                    prior=self._load(other)
                    selected=prior.get("reuse_key")==state["reuse_key"] and prior["status"]=="succeeded"
                    if prior["status"]=="succeeded" and meta["identity_policy"]=="staged-v1":
                        from application_reuse import matches,ReuseKeyRejected
                        try:
                            previous_meta=self._checked_task_metadata(other.parent.name)
                            selected=matches(prior,previous_meta,meta)
                        except ReuseKeyRejected as error:
                            raise PackageBindingRejected("cached_task_binding_rejected") from error
                        except (ValueError,TypeError,KeyError,OSError) as error:
                            if selected:raise PackageBindingRejected("cached_task_binding_rejected") from error
                            selected=False
                    if selected:
                        origin=prior["package"]["task_id"];package=self._dir(origin)/"program"
                        self._package_task_metadata(origin,meta,level)
                        binding=self.backend.check_package(package,level)
                        require(binding==prior["package"]["binding"],"Cached package changed")
                        control.check()
                        state.update(status="succeeded",phase="complete",failure=None,acceptance=prior["acceptance"],
                                     package=dict(task_id=origin,binding=binding,reused=True),generation=prior.get("generation"))
                        self._write(p/"state.json",state);return self.result(task)
                require(getattr(provider,"kind",None) in ("scripted_replay","deepseek_api","deepseek_offline"),
                        "Only the configured provider or explicit offline replay is supported")
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
                    candidate=None
                    try:
                        candidate=strict_json(raw)
                        static=validate_program(candidate,meta["request"])
                    except SyntaxError:
                        if policy == LEGACY:
                            raise
                        result=dict(status="failed",failure=dict(layer="source_parse"))
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
                            options={} if control.artifacts is None else dict(artifact_observer=control.register_artifact)
                            result=self.backend.qualify(meta["request"],candidate,level,**options)
                            control.check(force=True)
                    if policy == PRECISE:
                        result=attach(result,candidate=candidate,request=meta["request"])
                    last.clear();last.update(result)
                    control.last_failure=structured_failure(result,policy)
                    # Preserve full evidence locally; status() never returns it.
                    self._write(p/("validation-%d.json"%index),result)
                    return result
                def record(index,raw,feedback):
                    self._write(p/("response-%d.json"%index),dict(raw=raw,feedback=feedback))
                    control.save()
                loop=synthesize(meta["request"],provider,evaluate,record,max_repairs=b["generations"]-1,
                                validation_level=level,control=control)
                retained_loop=copy.deepcopy(loop)
                if provider.kind=="deepseek_api":
                    from generation_receipt import safe_metrics
                    retained_loop["provider_metrics"]=safe_metrics(loop["provider_metrics"])
                self._write(p/"loop.json",retained_loop)
                control.check()
                if loop["status"]=="passed":
                    state["phase"]="packaging";control.save()
                    export_options={}
                    if "model_provenance" in meta:
                        export_options["model_provenance"]=meta["model_provenance"]
                    if "model_source" in meta:
                        export_options["model_source"]=meta["model_source"]
                    if provider.kind=="deepseek_api":
                        require(meta["identity_policy"]=="staged-v1","Generation evidence needs staged identity")
                        from application_generation import capture
                        receipt,state["generation"]=capture(p,provider,loop,meta["request"],meta["identity"],state)
                        self._write(p/"generation-provenance.json",receipt)
                        export_options["generation_provenance"]=receipt
                    else:
                        state["generation"]=dict(provider=provider.kind,agent_generated=False,
                            generations=loop["attempts"],http_attempts=state["http_attempts"])
                    self.backend.export(last,p/"program",level,**export_options)
                    control.check(force=True)
                    binding=self.backend.check_package(p/"program",level)
                    require(self._identity_matches(meta),"Dependency changed during task")
                    control.check()
                    control.check(force=True)
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
                    terminal_reason={"provider_failed":"provider_failure","provider_exhausted":"provider_failure",
                        "integrity_failed":"integrity_rejected","infrastructure_failed":"environment_failure"}.get(loop["status"])
                    if policy == PRECISE and terminal_reason is not None:
                        from application_diagnostics import make
                        state["failure"]=make(terminal_reason)
                    else:
                        state["failure"]=control.last_failure or dict(code="provider_or_worker_failure",layer="provider")
            except Stopped as stop:state["status"]=stop.status
            except OperationalStopped as stop:
                signal.setitimer(signal.ITIMER_REAL,0)
                state.update(status=stop.status,failure=stop.public_failure,package=None,acceptance=None)
            except PackageBindingRejected as error:
                state.update(status="service_error",failure=error.public_failure(),package=None)
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
    def __init__(self,app,path,state,bounds,*,policy=LEGACY,operational_limits=None):
        require(policy in POLICIES,"Feedback policy")
        self.policy=policy
        from application_diagnostics import RepairDecision
        self.repair_decision=RepairDecision() if policy == PRECISE else None
        self.app=app;self.path=path;self.state=state;self.bounds=bounds
        self.deadline=time.monotonic()+bounds["wall_seconds"]
        self.seen=set();self.last_failure=None;self.last_key=None;self.repeated=0
        self.artifacts=None if operational_limits is None else ArtifactMonitor(app,path,operational_limits)
    def observe_failure(self,result):self.last_failure=structured_failure(result,self.policy)
    def save(self):
        if self.artifacts is not None:self.state["operational_usage"]=self.artifacts.usage()
        self.app._write(self.path/"state.json",self.state)
    def check(self,*,force=False):
        if (self.path/"cancel.json").exists():raise Stopped("cancelled")
        if time.monotonic()>=self.deadline:raise Stopped("timed_out")
        if self.artifacts is not None:
            self.artifacts.check(force=force)
            self.state["operational_usage"]=self.artifacts.usage()
    def register_artifact(self,path):
        self.check()
        self.artifacts.register(path)
        self.state["operational_usage"]=self.artifacts.usage()
        self.save()
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
        # Persist token usage without credentials or transport environment details.
        if metrics.get("provider")=="deepseek_api":
            from generation_receipt import safe_metrics
            self.app._write(self.path/"provider-metrics.json",safe_metrics(metrics))
            from application_transport_receipt import snapshot
            self.app._write(self.path/"provider-transport.json",snapshot(metrics))
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
        if self.repair_decision is not None:
            stop=self.repair_decision.before_candidate(raw)
            if stop:self.last_failure=self.repair_decision.last_failure
            return stop
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
        f=structured_failure(dict(failure=self.last_failure or {},public_feedback=feedback),self.policy)
        # Preserve trusted structured failure when available; never use its arbitrary diagnostic.
        if self.last_failure and self.last_failure.get("retry_policy"):f=structured_failure(dict(failure=self.last_failure),self.policy)
        self.last_failure=f;self.state["failure"]=f
        if self.repair_decision is not None:
            stop=self.repair_decision.after_failure(f)
            self.last_failure=self.repair_decision.last_failure
            self.state["failure"]=self.last_failure
            if stop:return stop
            self.state["phase"]="repairing";self.save()
            return
        route=decision(f)
        if route!="repair":return route
        key=repeat_key(f) if self.policy == PRECISE else digest({k:f.get(k) for k in ("layer","code","location","context")})
        self.repeated=0 if key is None else self.repeated+1 if key==self.last_key else 1
        self.last_key=key
        if self.repeated>=2:
            self.last_failure=dict(f,stop_reason="repeated_failure")
            self.state["failure"]=self.last_failure
            return "validation_failed"
        self.state["phase"]="repairing";self.save()
