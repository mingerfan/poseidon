"""Offline benchmark CLI; frozen model indices are separate from legacy batch manifests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from benchmark_graph import canonical, digest, samples, signature, validate
from benchmark_suite import ROOT, VERSION, QUOTAS, generate, summary
from benchmark_semantics import ledger, legacy_overlap

DEFAULT=ROOT/"scripts/baseline/benchmarks/semantic-v1-chunk-helpers-r29"


def dump(path,value):
    temporary=path.with_name(path.name+".partial")
    temporary.write_bytes(json.dumps(value,indent=2,sort_keys=True,allow_nan=False).encode()+b"\n")
    temporary.replace(path)


def source_hashes():
    paths=list((ROOT/"scripts/baseline").glob("benchmark_*.py"))+[ROOT/"scripts/benchmark.py",ROOT/"scripts/baseline/unified_graph_exercises.py",ROOT/"scripts/baseline/unified_public_exercises.py"]
    paths += [ROOT/'scripts/baseline'/name for name in ('upstream_helper_directed_cases.py','upstream_candidate_cases.py','upstream_bn_candidate_cases.py','upstream_adapters/scenarios.py','upstream_concat_cases.py','upstream_spatial_cases.py','upstream_spatial_mapped_cases.py','upstream_fused_cases.py','upstream_downsample_cases.py','upstream_virtual_cases.py','upstream_chunk_cases.py','rejection_benchmark.py','rejection_benchmark_cli.py','unified_chunk_cases.py','unified_chunk_layout.py')]
    import subprocess
    paths+=list((ROOT/"third_party/dacapo/python/poly/poly/data").glob("*"))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def build(directory):
    if directory.exists(): raise ValueError("Preserve existing release; choose a new directory")
    rows=generate(); s=summary(rows)
    if s["models"]!=1200 or s["unique_signatures"]!=1200 or s["topology_groups"]<300 or s["small_models"]<960:
        raise ValueError("Benchmark diversity/size gate")
    overlap=legacy_overlap(rows)
    if overlap["normalization_errors"]: raise ValueError("Legacy anchors failed normalization")
    if overlap["overlaps"]: raise ValueError("New corpus duplicates legacy anchors")
    directory.mkdir(parents=True)
    shards=[]
    for i in range(0,len(rows),48):
        name="shard-"+str(i//48).zfill(3)+".json"; content=dict(version=VERSION,rows=rows[i:i+48])
        dump(directory/name,content)
        shards.append(dict(file=name,sha256=hashlib.sha256((directory/name).read_bytes()).hexdigest(),cases=len(content["rows"])))
    coverage=ledger(rows);dump(directory/"coverage.json",coverage);dump(directory/"legacy-overlap.json",overlap)
    index=dict(**s,shards=shards,source_hashes=source_hashes(),
               coverage_sha256=hashlib.sha256((directory/"coverage.json").read_bytes()).hexdigest(),
               legacy_overlap_sha256=hashlib.sha256((directory/"legacy-overlap.json").read_bytes()).hexdigest(),
               reference_status="not_run",agent_status="not_run")
    dump(directory/"index.json",index)
    return s


def strict_file(path,limit):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>limit: raise ValueError("Benchmark file gate")
    def pairs(items):
        result={}
        for k,v in items:
            if k in result: raise ValueError("Duplicate JSON key")
            result[k]=v
        return result
    def nonfinite(x): raise ValueError("Nonfinite JSON")
    return json.loads(path.read_bytes(),object_pairs_hook=pairs,parse_constant=nonfinite)


def load(directory,check_sources=True):
    index=strict_file(directory/"index.json",4*1024**2)
    if index["version"]!=VERSION or not 1<=len(index["shards"])<=100: raise ValueError("Index bounds")
    if check_sources and index["source_hashes"]!=source_hashes(): raise ValueError("Frozen benchmark source changed")
    rows=[]; files=set()
    for shard in index["shards"]:
        name=shard["file"]
        if Path(name).name!=name or name in files: raise ValueError("Unsafe/duplicate shard")
        files.add(name);p=directory/name
        data=strict_file(p,1024**2)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=shard["sha256"]: raise ValueError("Shard hash")
        if data["version"]!=VERSION or not 1<=len(data["rows"])<=48 or len(data["rows"])!=shard["cases"]:
            raise ValueError("Shard count/version")
        rows.extend(data["rows"])
    for name,key in (("coverage.json","coverage_sha256"),("legacy-overlap.json","legacy_overlap_sha256")):
        p=directory/name;strict_file(p,4*1024**2)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=index[key]: raise ValueError("Ledger hash")
    groups={}
    for row in rows:
        g=row["model"]
        if digest(g)!=row["model_sha256"] or signature(g)!=row["signature"] or signature(g,True)!=row["topology"]:
            raise ValueError("Model identity changed")
        if row["topology"] in groups and groups[row["topology"]]!=row["split"]: raise ValueError("Split leakage")
        groups[row["topology"]]=row["split"]
    s=summary(rows)
    if s["models"]!=1200 or s["unique_signatures"]!=1200 or s["topology_groups"]<300 or s["small_models"]<960:
        raise ValueError("Benchmark release count/diversity")
    if any(s[k]!=index[k] for k in s): raise ValueError("Index summary mismatch")
    return rows,index


def ideal_reference(row,inputs):
    """Separate ideal operator, never treated as the polynomial's execution oracle."""
    import numpy as np
    spec=row["metadata"].get("ideal")
    if not spec: return None
    x=next(iter(inputs.values()))
    if spec["op"]=="relu": result=np.maximum(x,0)
    elif spec["op"]=="silu": result=x/(1+np.exp(-x))
    else:
        if spec["padding"]: x=np.pad(x,[(0,0)]*(x.ndim-1)+[(1,1)])
        if spec["negate"]: x=-x
        result=np.maximum(x[...,::2],x[...,1::2])
    return result


def plaintext(directory,out,resume=False):
    import numpy as np
    import torch
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as pytorch
    from platform_config import require_python_packages,identity
    require_python_packages(torch,np)
    torch.set_num_threads(2)
    rows,index=load(directory)
    binding=digest(dict(index_sha256=hashlib.sha256((directory/"index.json").read_bytes()).hexdigest(),
                        source_hashes=source_hashes(),mode="plaintext",probes=16,
                        reference_atol=1e-12,reference_rtol=1e-12))
    if out.exists():
        if not resume: raise ValueError("Preserve results: use --resume with identical frozen task")
        prior=strict_file(out/"run.json",1024**2)
        if prior["binding"]!=binding: raise ValueError("Resume binding mismatch")
    else:
        out.mkdir(parents=True)
        dump(out/"run.json",dict(binding=binding,mode="plaintext",platform_identity=identity(),
             versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__),
             frozen_index=index,started=time.time()))
    checkpoint=out/"checkpoint.json"
    hashes=strict_file(checkpoint,1024**2) if checkpoint.exists() else {}
    results=[]
    for row in rows:
        file=out/(row["model"]["id"]+".json")
        if file.exists():
            if hashes.get(file.name)!=hashlib.sha256(file.read_bytes()).hexdigest():
                raise ValueError("Resume result hash mismatch or uncheckpointed result")
            previous=strict_file(file,1024**2)
            if previous.get("status") not in ("passed","failed") or previous["binding"]!=binding or previous["model_sha256"]!=row["model_sha256"]:
                raise ValueError("Resume row changed")
            # Only a terminal, intact record can be reused. No old success is inferred.
            result=previous
        else:
            result=dict(id=row["model"]["id"],binding=binding,model_sha256=row["model_sha256"],
                        status="running",tests=0,max_abs_error=0.0,reference_outputs_sha256=[],
                        encrypted_execution=False,agent_calls=0)
            start=time.monotonic()
            try:
                for inputs in samples(row["model"]):
                    expected=mathematical(row["model"],inputs);actual=pytorch(row["model"],inputs)
                    for name in expected:
                        if expected[name].shape!=actual[name].shape or not np.isfinite(actual[name]).all():
                            raise ValueError("Independent reference shape/nonfinite mismatch")
                        np.testing.assert_allclose(actual[name],expected[name],atol=1e-12,rtol=1e-12)
                        result["max_abs_error"]=max(result["max_abs_error"],float(np.max(np.abs(actual[name]-expected[name]))))
                    result["reference_outputs_sha256"].append(digest({n:v.tolist() for n,v in expected.items()}))
                    ideal=ideal_reference(row,inputs)
                    if ideal is not None:
                        value=next(iter(expected.values()))
                        result["ideal_function_max_abs_difference"]=max(result.get("ideal_function_max_abs_difference",0),
                                                                          float(np.max(np.abs(value-ideal))))
                    result["tests"]+=1
                result["status"]="passed"
            except Exception as error:
                result.update(status="failed",failure_layer="model_reference",error=str(error)[:2000])
            result["seconds"]=time.monotonic()-start
            dump(file,result)
            hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
            dump(checkpoint,hashes)
        results.append(result)
    report=dict(binding=binding,planned=len(rows),passed=sum(x["status"]=="passed" for x in results),
        failed=sum(x["status"]=="failed" for x in results),skipped=0,
        tests=sum(x["tests"] for x in results),max_reference_error=max(x["max_abs_error"] for x in results),
        result_hashes={x["id"]:hashlib.sha256((out/(x["id"]+".json")).read_bytes()).hexdigest() for x in results},
        seconds=sum(x["seconds"] for x in results),encrypted_execution=False,agent_calls=0,
        approximation_is_not_execution_error=True)
    dump(out/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="result_hashes"},indent=2))
    return int(report["failed"]>0)


def plan(directory):
    rows,index=load(directory)
    from benchmark_bridge import to_legacy
    counts={}; reasons={}
    for row in rows:
        try: to_legacy(row["model"]);status="legacy_candidate_compatible"
        except (ValueError,KeyError,TypeError) as e:
            status="requires_unified_backend";reasons[str(e)]=reasons.get(str(e),0)+1
        counts[status]=counts.get(status,0)+1
    return dict(**summary(rows),backend_preflight=counts,blockers=reasons,
                direct_helper_calls_validated=0,directed_tasks=len(strict_file(directory/"coverage.json",4*1024**2)["directed_tasks"]),
                live_authorized=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("plan","build","check","plaintext"))
    parser.add_argument("--suite",type=Path,default=DEFAULT)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.mode=="build": result=build(args.suite)
    elif args.mode=="check": result=summary(load(args.suite)[0])
    elif args.mode=="plan": result=plan(args.suite)
    else:
        if args.output is None: parser.error("plaintext needs --output in the execution work root")
        from workspace_paths import RESULTS
        if not args.output.resolve().is_relative_to(RESULTS.resolve()):
            parser.error("--output must be within the selected platform results directory")
        if not args.inside:
            import shlex
            from hecate_python_env import enter_nix,VENV
            command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/benchmark.py"),*sys.argv[1:],"--inside"]
            return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(command),seconds=1800)
        from hecate_python_env import VENV
        if not os.environ.get("IN_NIX_SHELL") or Path(sys.prefix)!=VENV:parser.error("Locked Nix Python required")
        return plaintext(args.suite,args.output,args.resume)
    print(json.dumps(result,indent=2))
    return 0
