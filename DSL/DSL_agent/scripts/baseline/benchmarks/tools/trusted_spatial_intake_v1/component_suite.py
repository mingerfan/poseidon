"""Staged spatial intake acceptance; independent plaintext and provenance only."""
import sys,json,shlex,time,hashlib,traceback,copy,resource,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
if "--inside"not in sys.argv:
    cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
      [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(cmd,seconds=240,environment="poseidon-cpu"))
sys.path.insert(0,str(HERE/"component-stage"));sys.path.insert(1,str(HERE))
import torch,numpy as np
from fixtures import cases
from spatial_component_fixtures import contexts
from pytorch_weight_package import export_state,read,binding
from pytorch_input_binding import CALL_FORMAT
from pytorch_public_expression_capture import prepare,wrapped_metadata,make_reference,check_reference
from pytorch_ordered_contract import validate_package,capture_binding,package_version,source_identity
from benchmark_graph import digest
from candidate_bundle import check_provenance
from hecate_python_env import WORK
torch.set_num_threads(1)
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(150,150))
out=WORK/"results"/("agent-spatial-source-component-stage-"+str(time.time_ns()));out.mkdir()
rows=[];positive=[];maximum=0;groups=values=0;start=time.monotonic()
definitions=(HERE/"fixtures.py").read_text()+"\n"
selections=[(n,m,i,o,dict(format=CALL_FORMAT,args=[dict(kind="tensor",name=x["name"])for x in i],kwargs={}),"strict-float64-v1")
            for n,m,i,o in cases()]+contexts()
first=None
for name,module,inputs,outputs,call,dtype in selections:
    folder=out/name;folder.mkdir();source=folder/"source";source.mkdir()
    weights=export_state(module,source,resource_profile="general-tensors-v1")
    arrays,raw=read(source,weights,resource_profile="general-tensors-v1");weights_binding=binding(arrays,weights)
    # Repository fixtures are trusted test objects; the source metadata is still strictly bound.
    fixture=(HERE/"fixtures.py").read_text()
    extra=(HERE/"spatial_component_fixtures.py").read_text()
    files={"fixtures.py":fixture,"spatial_component_fixtures.py":extra,"models.py":"from fixtures import cases\nfrom spatial_component_fixtures import contexts\ndef build():\n    return next(x[1] for x in [*cases(),*contexts()] if x[0]=="+repr(name)+")\n"}
    for file,text in files.items():(source/file).write_text(text)
    spec=dict(format="poseidon-trusted-pytorch-v27",entry="models.build",factory=dict(args=[],kwargs={}),
      files={file:hashlib.sha256(text.encode()).hexdigest()for file,text in files.items()},weights=weights,
      inputs=inputs,outputs=outputs,dtype_policy=dtype,
      approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},nodes=[],modules=[]),call=call)
    row=dict(case=name,status="running");rows.append(row)
    try:
        validate_package(spec,files)
        capture=folder/"capture";capture.mkdir()
        record=prepare(module,spec,capture,expected_weight_binding=weights_binding,resource_profile="general-tensors-v1")
        metadata=wrapped_metadata(record)
        report=make_reference(record,spec,metadata,resource_profile="general-tensors-v1",expected_weight_arrays=arrays)
        check_reference(report,record["high_graph"],record["lowered_graph"],spec,resource_profile="general-tensors-v1",
            expected_weight_binding=weights_binding,expected_weight_arrays=arrays)
        lower,decomp=capture_binding(record["high_graph"],spec,resource_profile="general-tensors-v1",
          capture_metadata=metadata,expected_weight_binding=weights_binding,expected_weight_arrays=arrays)
        provenance=dict(original=record["high_graph"],decomposition=decomp,reference_report=report,
          trusted_pytorch=dict(format=package_version(spec),manifest=spec,files=files,
                               weight_files=__import__("pytorch_ordered_contract").encode_weight_files(spec,raw,resource_profile="general-tensors-v1")))
        check_provenance(provenance,lower)
        (folder/"provenance.json").write_text(json.dumps(provenance,allow_nan=False))
        (source/"manifest.json").write_text(json.dumps(spec,allow_nan=False))
        finite=record["original_reference"]
        maximum=max(maximum,finite["max_original_torch_vs_independent_ideal"],finite["max_original_vs_normalized_torch"],finite["max_decomposition_error"])
        groups+=16;values+=finite["compared_values"]
        row.update(status="passed",groups=16,values=finite["compared_values"],
            maximum_difference=max(finite["max_original_torch_vs_independent_ideal"],finite["max_decomposition_error"]),
            manifest_sha256=digest(spec),high_sha256=digest(record["high_graph"]),lowered_sha256=digest(lower))
        positive.append(row)
        if first is None:first=(module,spec,arrays,weights_binding,metadata,record)
    except Exception as error:
        row.update(status="failed",error_type=type(error).__name__,error=str(error))
        (folder/"failure.txt").write_text(traceback.format_exc())
    print(json.dumps(row),flush=True)
modules=["test_pytorch_input_binding","test_pytorch_public_dtype_cast","test_pytorch_closed_public_arithmetic",
         "test_operator_spatial_graph","test_unified_public_arithmetic"]
suite=unittest.defaultTestLoader.loadTestsFromNames(modules)
with(out/"compatibility-tests.log").open("x")as stream:
    unit=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
report=dict(format="poseidon-spatial-source-component-stage-v1",passed=len(positive),
    failed=sum(r["status"]=="failed"for r in rows),rows=rows,plaintext_groups=groups,plaintext_values=values,
    maximum_difference=maximum,unit_tests=dict(run=unit.testsRun,failed=len(unit.failures),errors=len(unit.errors),skipped=len(unit.skipped)),
    elapsed_seconds=time.monotonic()-start,source_identity=source_identity(),result=str(out),
    production_registered=False,api_calls=0,compilations=0,encrypted_calls=0)
report["binding"]=digest(report);(out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in report.items()if k not in("rows","source_identity")}),flush=True)
raise SystemExit(1 if report["failed"]or not unit.wasSuccessful()else 0)
