"""Isolated observations of hash-bound original PyTorch module inputs.

This is an explicit calibration tool. It reuses existing source, weight,
runtime and sandbox checks. It does not fit, alter approximations, execute
DSL/FHE, or import the source model into the host.
"""
import hashlib,json,os,signal,subprocess,sys
from pathlib import Path
from source_calibration_contract import CalibrationError,require,checked
from source_precision_reference import (
    prepare_source,checked_samples,implementation_paths as retained_paths,
    SourceReferenceError)
FORMAT="poseidon-source-calibration-report-v1"

def implementation_paths():
    paths=retained_paths()
    for name in ("source_calibration.py","source_calibration_contract.py","source_calibration_observer.py","model_output_contract.py"):
        paths[name]=Path(__file__).with_name(name)
    return paths
def implementation():
    return {name:hashlib.sha256(path.read_bytes()).hexdigest()for name,path in sorted(implementation_paths().items())}

def check_report(report,request):
    from benchmark_graph import digest
    require(type(report)is dict and report.get("format")==FORMAT and report.get("status")=="observed"and
        report.get("request_sha256")==digest(request)and report.get("implementation")==request["implementation"]and
        report.get("selection_sha256")==digest(request["selection"])and
        report.get("manifest_sha256")==digest(request["manifest"])and report.get("runtime")==request["runtime"]and
        report.get("input_dtypes")==request["input_dtypes"]and
        type(report.get("samples"))is int and report["samples"]==len(request["samples"])and
        report.get("purpose")=="calibration"and report.get("model_imported_in_host")is False and
        report.get("native_dtype_preserved")is True and report.get("encrypted_execution")is False and
        report.get("approximation_modified")is False and report.get("reference_modified")is False and
        report.get("application_accuracy_certified")is False and
        type(report.get("api_calls"))is int and report["api_calls"]==0,
        "calibration_worker_protocol_error")
    require(report.get("checks")==["registered_state_unchanged","rng_unchanged","input_storage_isolated",
        "observer_output_equal_to_original","repeat_observations_equal","hooks_removed"],
        "calibration_worker_protocol_error")
    sandbox=report.get("sandbox")
    require(type(sandbox)is dict and sandbox.get("network_namespace")!=request["host_net_namespace"]and
        sandbox==dict(network_isolated=True,network_namespace=sandbox.get("network_namespace"),
            address_space_bytes=4*1024**3,cpu_seconds=150,file_bytes=16*1024**2,open_files=128),
        "calibration_worker_protocol_error")
    observations=report.get("observations")
    require(type(observations)is dict and set(observations)=={s["id"]for s in request["selection"]["selectors"]},
        "calibration_worker_protocol_error")
    expected={"module","argument","projection","samples","calls","values","minimum","maximum","mean",
        "std_population","dtypes","shapes","declared_domain","outside_domain_calls","domain_status"}
    import math
    for selector in request["selection"]["selectors"]:
        item=observations[selector["id"]]
        require(type(item)is dict and set(item)==expected and
            all(item[key]==selector[key]for key in ("module","argument","projection"))and
            item["samples"]==report["samples"]and type(item["calls"])is int and item["calls"]>=item["samples"]and
            type(item["values"])is int and item["values"]>=item["calls"]and
            all(type(item[key])in(int,float)and math.isfinite(item[key])for key in
                ("minimum","maximum","mean","std_population"))and
            item["minimum"]<=item["mean"]<=item["maximum"]and item["std_population"]>=0 and
            type(item["dtypes"])is list and item["dtypes"]and all(x in ("float32","float64")for x in item["dtypes"])and
            type(item["shapes"])is list and item["shapes"]and all(type(s)is list and len(s)<=32 and
                all(type(x)is int and x>0 for x in s)for s in item["shapes"])and
            item["declared_domain"]==selector["domain"]and type(item["outside_domain_calls"])is int and
            0<=item["outside_domain_calls"]<=item["calls"]and
            item["domain_status"]==("not_declared"if selector["domain"]is None else
             "outside"if item["outside_domain_calls"]else "inside"),"calibration_worker_protocol_error")
    require(type(report.get("binding"))is str and report["binding"]==digest({k:v for k,v in report.items()if k!="binding"}),
        "calibration_worker_protocol_error")
    return report

def run(manifest_path,model,samples,output,*,expected_manifest_sha256,selection,
        expected_selection_sha256,input_dtypes,output_structure,
        resource_profile="general-tensors-v1",trusted=False):
    require(trusted is True)
    from benchmark_graph import digest
    from strict_json_files import strict_file
    from model_weight_package import regular_bytes
    from pytorch_weight_package import read
    from pytorch_ordered_contract import SOURCE_FORMATS
    from pytorch_source_contract import STATE_FORMATS
    from trusted_pytorch_import import runtime_identity
    from model_resources import resolve
    from hecate_python_env import VENV,WORK
    from workspace_paths import require_linux_backend,work_root
    from platform_config import configuration
    from process_resource_limits import limits
    require_linux_backend();profile=resolve(resource_profile)
    selection=checked(selection)
    require(digest(selection)==expected_selection_sha256,"calibration_binding_changed")
    path=Path(manifest_path)
    try:
        spec,paths=prepare_source(path)
        require(digest(spec)==expected_manifest_sha256,"calibration_binding_changed")
        wire,dtypes,tree,shapes=checked_samples(spec,model,samples,input_dtypes,output_structure,resource_profile)
        weights={}
        if spec["format"]in(*STATE_FORMATS,*SOURCE_FORMATS):
            _,weights=read(path.parent,spec["weights"],resource_profile=resource_profile)
    except CalibrationError:raise
    except (SourceReferenceError,ValueError,TypeError,KeyError,OSError):
        raise CalibrationError("calibration_input_rejected")from None
    output=Path(output)
    require(output.is_absolute()and output==output.resolve()and output.parent.is_relative_to(WORK/"results")and
        not output.exists()and not output.is_symlink())
    library=os.environ.get("HECATE_PYTHON_LIBRARY_PATH")
    require(type(library)is str and library and
        all(Path(v).is_relative_to("/nix/store")for v in library.split(":")),"calibration_environment_unavailable")
    identity=implementation()
    source_files={name:regular_bytes(p,65536)for name,p in paths.items()}
    require(not set(paths).intersection(implementation_paths()))
    output.mkdir(mode=0o700);snap=output/"source-snapshot";snap.mkdir(mode=0o700)
    for name,raw in {**source_files,**weights}.items():
        with (snap/name).open("xb")as f:f.write(raw)
    request=dict(format=FORMAT,manifest=spec,samples=wire,input_dtypes=dtypes,output_structure=tree,
        output_shapes=shapes,resource_profile=resource_profile,tensor_elements=profile.tensor_elements,
        selection=selection,runtime=runtime_identity(),implementation=identity,
        host_net_namespace=os.readlink("/proc/self/ns/net"))
    payload=output/"request.json"
    with payload.open("x")as f:json.dump(request,f,allow_nan=False)
    command=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL","--clearenv",
        "--ro-bind","/nix/store","/nix/store","--ro-bind",str(VENV),str(VENV),
        "--ro-bind","/sys/devices/system/cpu","/sys/devices/system/cpu",
        "--proc","/proc","--dev","/dev","--tmpfs","/tmp","--dir","/app",
        "--ro-bind",str(snap),"/model","--ro-bind",str(payload),"/request.json",
        "--bind",str(output),"/out"]
    for name,p in implementation_paths().items():command+=["--ro-bind",str(p),"/app/"+name]
    environment=dict(PYTHONPATH="/app:/model",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
        LD_LIBRARY_PATH=library,HOME="/nonexistent",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",LANG="C.UTF-8",POSEIDON_PLATFORM=configuration()["id"],
        POSEIDON_WORK_ROOT=str(work_root()))
    for name,value in environment.items():command+=["--setenv",name,value]
    command+=["--chdir","/out","--",str(VENV/"bin/python"),"-B","/app/source_calibration.py","--worker"]
    with (output/"forward.log").open("x")as log:
        process=subprocess.Popen(command,env={"PATH":"/usr/bin:/bin"},stdout=log,stderr=subprocess.STDOUT,
            preexec_fn=limits,start_new_session=True)
        try:process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise CalibrationError("calibration_timeout")from None
    if process.returncode:
        failed=output/"calibration-failure.json"
        if failed.exists():
            item=strict_file(failed,131072)
            require(item.get("request_sha256")==digest(request)and
                item.get("binding")==digest({k:v for k,v in item.items()if k!="binding"}),
                "calibration_worker_protocol_error")
            raise CalibrationError(item["code"])
        raise CalibrationError("calibration_source_failed")
    require(implementation()==identity,"calibration_binding_changed")
    current,now=prepare_source(path)
    require(current==spec and all(regular_bytes(now[n],65536)==raw for n,raw in source_files.items()),
        "calibration_binding_changed")
    for name,raw in weights.items():
        require(regular_bytes(path.parent/name,len(raw))==raw,"calibration_binding_changed")
    report=strict_file(output/"calibration.json",16*1024**2)
    return check_report(report,request)

def worker():
    import importlib,random,resource
    import numpy as np
    import torch
    from benchmark_graph import digest
    from strict_json_files import strict_file
    from trusted_pytorch_import import validate_package as legacy_validate
    from pytorch_ordered_contract import SOURCE_FORMATS,allows_none_output,shim,validate_package
    from pytorch_source_contract import STATE_FORMATS,factory_parameters,output_tree
    from pytorch_weight_package import read,load_state
    from source_calibration_observer import ModuleObserver,summarize
    request=strict_file(Path("/request.json"),16*1024**2);spec=request["manifest"]
    require(request["format"]==FORMAT,"calibration_worker_protocol_error")
    selection=checked(request["selection"])
    runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=str(torch.__version__),
        cuda=torch.version.cuda,hip=getattr(torch.version,"hip",None))
    require(runtime==request["runtime"]and implementation()==request["implementation"],"calibration_binding_changed")
    files={n:Path("/model",n).read_text()for n in spec["files"]}
    if spec["format"]in SOURCE_FORMATS:
        validate_package(spec,files);parameters=factory_parameters(shim(spec))
    else:
        legacy_validate(spec,files);parameters=factory_parameters(spec)
    torch.set_num_threads(1)
    module_name,entry=spec["entry"].split(".")
    module=getattr(importlib.import_module(module_name),entry)(*parameters["args"],**parameters["kwargs"])
    require(isinstance(module,torch.nn.Module))
    if spec["format"]in(*STATE_FORMATS,*SOURCE_FORMATS):
        arrays,_=read("/model",spec["weights"],resource_profile=request["resource_profile"])
        load_state(module,arrays,spec["weights"])
    module=module.cpu().eval()
    before={n:v.detach().clone()for n,v in module.state_dict().items()}
    rng=torch.get_rng_state().clone();py_rng=random.getstate();np_rng=np.random.get_state()
    def unchanged():
        now=module.state_dict()
        require(set(now)==set(before)and all(now[n].dtype==v.dtype and torch.equal(now[n],v)for n,v in before.items()),
                "calibration_source_failed")
        require(torch.equal(torch.get_rng_state(),rng)and random.getstate()==py_rng and
            all(np.array_equal(a,b)for a,b in zip(np.random.get_state(),np_rng)),"calibration_source_failed")
    def forward(sample):
        tensors=[]
        for spec_input in spec["inputs"]:
            name=spec_input["name"];array=np.asarray(sample[name],dtype=request["input_dtypes"][name])
            require(list(array.shape)==spec_input["shape"]and np.isfinite(array).all())
            tensors.append(torch.from_numpy(array.copy()))
        args=tensors;kwargs={}
        if "call"in spec:
            from pytorch_input_binding import materialize,validate_call
            validate_call(spec["call"],spec["inputs"])
            by_name={item["name"]:value for item,value in zip(spec["inputs"],tensors)}
            args=[materialize(item,by_name)for item in spec["call"]["args"]]
            kwargs={name:materialize(value,by_name)for name,value in spec["call"]["kwargs"].items()}
        with torch.no_grad():
            raw,tree=output_tree(module(*args,**kwargs),lambda v:isinstance(v,torch.Tensor),spec["outputs"],
                scalar_tensor=lambda v:v.ndim==0,allow_none=allows_none_output(spec))
        require(tree==request["output_structure"],"calibration_source_failed")
        values={}
        for name,value in zip(spec["outputs"],raw):
            require(value.device.type=="cpu"and value.layout==torch.strided and
                value.dtype in(torch.float32,torch.float64),"calibration_source_failed")
            array=value.detach().numpy().copy()
            if array.ndim==0:array=array.reshape(1)
            require(list(array.shape)==request["output_shapes"][name]and np.isfinite(array).all(),"calibration_source_failed")
            values[name]=array
        unchanged();return values
    observations=[];output_hashes=[]
    observer=ModuleObserver(module,selection,tensor_elements=request["tensor_elements"])
    for sample in request["samples"]:
        baseline=forward(sample)
        with observer:
            observer.begin();one=forward(sample);first=observer.observations()
            observer.begin();two=forward(sample);second=observer.observations()
        require(all(np.array_equal(baseline[n],one[n])and np.array_equal(one[n],two[n])for n in baseline),
                "calibration_observer_changed_forward")
        require(first==second,"calibration_observation_nondeterministic")
        observations.append(first);output_hashes.append(digest({n:v.tolist()for n,v in baseline.items()}))
    namespace=os.readlink("/proc/self/ns/net")
    require(namespace!=request["host_net_namespace"]and
        resource.getrlimit(resource.RLIMIT_AS)==(4*1024**3,4*1024**3)and
        resource.getrlimit(resource.RLIMIT_CPU)==(150,155)and
        resource.getrlimit(resource.RLIMIT_FSIZE)==(16*1024**2,16*1024**2)and
        resource.getrlimit(resource.RLIMIT_NOFILE)==(128,128),"calibration_worker_protocol_error")
    report=dict(format=FORMAT,status="observed",purpose="calibration",request_sha256=digest(request),
        implementation=request["implementation"],selection_sha256=digest(selection),manifest_sha256=digest(spec),
        runtime=runtime,input_dtypes=request["input_dtypes"],samples=len(observations),
        observations=summarize(selection,observations),forward_output_sha256=output_hashes,
        native_dtype_preserved=True,model_imported_in_host=False,encrypted_execution=False,
        approximation_modified=False,reference_modified=False,application_accuracy_certified=False,api_calls=0,
        checks=["registered_state_unchanged","rng_unchanged","input_storage_isolated",
            "observer_output_equal_to_original","repeat_observations_equal","hooks_removed"],
        sandbox=dict(network_isolated=True,network_namespace=namespace,address_space_bytes=4*1024**3,
            cpu_seconds=150,file_bytes=16*1024**2,open_files=128))
    report["binding"]=digest(report)
    with Path("/out/calibration.json").open("x")as f:json.dump(report,f,allow_nan=False,indent=2)

if __name__=="__main__":
    require(sys.argv[1:]==["--worker"])
    try:worker()
    except Exception as error:
        from benchmark_graph import digest
        request=json.loads(Path("/request.json").read_bytes())
        code=error.code if isinstance(error,CalibrationError)else "calibration_source_failed"
        record=dict(format="poseidon-source-calibration-failure-v1",request_sha256=digest(request),code=code)
        record["binding"]=digest(record)
        with Path("/out/calibration-failure.json").open("x")as f:json.dump(record,f)
        raise
