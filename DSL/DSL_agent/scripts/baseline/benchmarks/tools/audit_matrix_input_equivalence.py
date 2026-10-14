"""Compare paired JSON matrix graphs and isolated trusted PyTorch captures.

This development audit performs no DSL generation, compilation or encrypted execution.
It checks current admission independently of the mathematical reference comparisons.
"""
import argparse
import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

OPTIONS = dict(chunk_period=8,capacity_packing_version="capacity-public-tensors-v9",
               generation_guidance="explicit-v10",public_weight_policy="descriptors-v1")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def preserve_binding(root, path):
    if path is None:
        return None
    data=json.loads(Path(path).read_text())
    for key in ("source_hashes","case_hashes","tool_hashes"):
        for name, expected in data[key].items():
            item=Path(name)
            item=item if item.is_absolute() else root/item
            if sha(item)!=expected:
                raise ValueError("Frozen "+key+" changed: "+name)
    return dict(binding=data["binding"],source_files=len(data["source_hashes"]),
                case_files=len(data["case_hashes"]),tool_files=len(data["tool_hashes"]),matched=True)

def rejection(call):
    try:
        call()
    except (ValueError,TypeError,KeyError,IndexError,OverflowError):
        return True
    return False

def run(cases, output, *, frozen_binding=None, seconds=600):
    import numpy as np
    import torch
    from application_capabilities import assess_model
    from benchmark_graph import samples,digest
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as tensor_reference
    from operator_reference import evaluate as high_reference
    from pytorch_matrix_plan import check_record
    from pytorch_weight_package import read
    from trusted_pytorch_import import run as capture,runtime_identity
    from hecate_python_env import ROOT,WORK

    cases,output=Path(cases).resolve(),Path(output).resolve()
    if output.exists() or not output.parent.is_relative_to(WORK/"results"):
        raise ValueError("Fresh audit directory under results required")
    if seconds<180 or seconds>1200:
        raise ValueError("Audit time bound")
    output.mkdir()
    started=time.monotonic()
    initial=preserve_binding(ROOT,frozen_binding)
    inventory={str(p.relative_to(cases)):sha(p) for p in cases.rglob("*") if p.is_file()}
    owned_paths=[
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("generate_matrix_entry_sources.py"),
        Path(__file__).resolve().with_name("test_matrix_entry_audit.py"),
    ]
    audit_sources={str(p.relative_to(ROOT)):sha(p) for p in owned_paths}
    catalog=json.loads((cases/"catalog.json").read_text())
    if catalog["planned"]!=24 or catalog["partitions"]!=8 or len(catalog["cases"])!=24:
        raise ValueError("Cross-input fixture inventory")
    runtime=runtime_identity()
    if runtime!=dict(python=[3,10,14],numpy="1.25.2",torch="2.0.1",cuda=None,hip=None):
        raise ValueError("Pinned ARM CPU audit runtime")
    torch.set_num_threads(1)
    audit_binding=digest(dict(fixture_hashes=inventory,audit_source_hashes=audit_sources,
                             frozen_runtime_binding=initial,runtime=runtime,options=OPTIONS))
    rows=[]
    groups=negative_checks=0
    max_error=0.0
    def write():
        report=dict(format="poseidon-matrix-input-audit-v1",cases=rows,
            planned=len(catalog["cases"]),completed=len(rows),
            isolated_capture_passed=sum(x["capture_status"]=="passed" for x in rows),
            independent_reference_passed=sum(x["reference_status"]=="passed" for x in rows),
            direct_json_admitted=sum(x["direct_json_admission"]=="accepted_pending_validation" for x in rows),
            normalized_pytorch_admitted=sum(x.get("normalized_admission")=="accepted_pending_validation" for x in rows),
            parity_gaps=[x["id"] for x in rows if x["direct_json_admission"]!="accepted_pending_validation"
                        and x.get("normalized_admission")=="accepted_pending_validation"],
            negative_integrity_checks=negative_checks,independent_reference_groups=groups,
            independent_reference_tolerance=dict(atol=1e-12,rtol=1e-12),
            max_reference_absolute_error=max_error,
            seconds=time.monotonic()-started,runtime=runtime,
            fixture_hashes=inventory,frozen_runtime_binding=initial,
            audit_source_hashes=audit_sources,audit_binding_sha256=audit_binding,
            admission_options=OPTIONS,
            old_frozen_benchmark_changed=False,existing_runtime_source_changed=False,
            compiled=False,encrypted_execution=False,agent_generated=False,
            paid_calls=0,installation=False,download=False,backend_changed=False,
            resource_profile="general-tensors-v1")
        (output/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
        return report
    for case in catalog["cases"]:
        if time.monotonic()-started>seconds-180:
            break
        preserve_binding(ROOT,frozen_binding)
        folder=cases/Path(case["manifest"]).parent
        high=json.loads((folder/"operator-model.json").read_text())
        if digest(high)!=case["operator_model_sha256"]:
            raise ValueError("Original mathematical graph identity")
        direct=assess_model(high,resource_profile="general-tensors-v1",options=OPTIONS)
        row=dict(id=case["id"],partition=case["partition"],context=case["context"],
                 first_shape=case["first_shape"],second_shape=case["second_shape"],
                 output_shape=case["output_shape"],public_side=case["public_side"],
                 original_operator_sha256=digest(high),
                 direct_json_admission=direct["admission"],direct_json=direct,
                 capture_status="not_run",reference_status="not_run")
        rows.append(row)
        capture_dir=output/(case["id"]+"-capture")
        try:
            captured=capture(folder/"manifest.json",capture_dir,
                             resource_profile="general-tensors-v1",trusted=True)
            row["capture_status"]="passed"
            row["capture_result"]=captured
        except (ValueError,TypeError,KeyError,IndexError,OverflowError) as error:
            row["capture_status"]="failed"
            row["failure_layer"]="trusted_pytorch_capture"
            row["failure"]=str(error)[:2000]
            write()
            continue
        core=json.loads((capture_dir/"model.json").read_text())
        imported=json.loads((capture_dir/"original.json").read_text())
        reference=json.loads((capture_dir/"reference-report.json").read_text())
        normalized=assess_model(core,options=OPTIONS)
        row.update(normalized_admission=normalized["admission"],normalized=normalized,
                   normalized_operator_sha256=digest(imported),core_sha256=digest(core),
                   worker_reference_groups=reference["probes"],
                   worker_max_error=reference["max_decomposition_error"],
                   capture_record_sha256=digest(reference["capture_metadata"]))
        local_max=0.0
        try:
            for inputs in samples(high,16):
                expected=high_reference(high,inputs)
                actual=mathematical(core,inputs,resource_profile="general-tensors-v1")
                independent=tensor_reference(core,inputs,resource_profile="general-tensors-v1")
                imported_math=high_reference(imported,inputs)
                operands=[np.asarray(high["constants"][name],dtype=np.float64)
                          if name in high["constants"] else inputs[name]
                          for name in high["nodes"][0]["inputs"]]
                torch_original=torch.matmul(*(torch.from_numpy(x) for x in operands)).numpy()
                for name, truth in expected.items():
                    for result in (actual[name],independent[name],imported_math[name],torch_original):
                        if result.shape!=truth.shape:
                            raise ValueError("Cross-input result shape differs")
                        np.testing.assert_allclose(result,truth,atol=1e-12,rtol=1e-12)
                        local_max=max(local_max,float(np.max(np.abs(result-truth))))
                groups+=1
            row["reference_status"]="passed"
            row["reference_groups"]=16
            row["max_reference_absolute_error"]=local_max
            max_error=max(max_error,local_max)
        except (ValueError,TypeError,KeyError,IndexError,AssertionError) as error:
            row["reference_status"]="failed"
            row["failure_layer"]="independent_reference"
            row["failure"]=str(error)[:2000]
        record=reference["capture_metadata"]["matrix_capture"]
        check_record(record,imported,resource_profile="general-tensors-v1")
        rejected=0
        changed=copy.deepcopy(record)
        changed["graph_sha256"]="0"*64
        rejected+=rejection(lambda:check_record(changed,imported,resource_profile="general-tensors-v1"))
        changed=copy.deepcopy(record)
        changed["entries"][0]["plan"]["result_shape"]=[7]
        rejected+=rejection(lambda:check_record(changed,imported,resource_profile="general-tensors-v1"))
        changed=copy.deepcopy(record)
        changed["entries"][0]["plan"]["public_side"]=(
            "right" if changed["entries"][0]["plan"]["public_side"]!="right" else "left")
        rejected+=rejection(lambda:check_record(changed,imported,resource_profile="general-tensors-v1"))
        if rejected!=3:
            raise ValueError("Altered matrix provenance was accepted")
        row["matrix_integrity_rejections"]=rejected
        negative_checks+=rejected
        if case["public_side"]:
            manifest=json.loads((folder/"manifest.json").read_text())
            mutated=output/(case["id"]+"-bad-weight")
            mutated.mkdir()
            for item in manifest["weights"].values():
                raw=(folder/item["file"]).read_bytes()
                target=mutated/item["file"]
                target.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            if not rejection(lambda:read(mutated,manifest["weights"],resource_profile="general-tensors-v1")):
                raise ValueError("Altered public weight was accepted")
            row["weight_integrity_rejections"]=1
            negative_checks+=1
        write()
        print(json.dumps(dict(id=row["id"],capture=row["capture_status"],reference=row["reference_status"],
            direct_json=row["direct_json_admission"],normalized=row["normalized_admission"])),flush=True)
    for name,expected in inventory.items():
        if sha(cases/name)!=expected:
            raise ValueError("Fixture changed during audit")
    if {str(p.relative_to(ROOT)):sha(p) for p in owned_paths}!=audit_sources:
        raise ValueError("Audit implementation changed during execution")
    final=preserve_binding(ROOT,frozen_binding)
    if final!=initial:
        raise ValueError("Frozen binding inventory changed")
    result=write()
    result["completed_within_window"]=result["completed"]==result["planned"] and result["seconds"]<=seconds
    result["source_binding_after"]=final
    (output/"report.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--preserve-binding",type=Path)
    parser.add_argument("--seconds",type=int,default=600)
    args=parser.parse_args()
    result=run(args.cases,args.output,frozen_binding=args.preserve_binding,seconds=args.seconds)
    print(json.dumps({k:result[k] for k in ("planned","completed","isolated_capture_passed",
        "independent_reference_passed","direct_json_admitted","normalized_pytorch_admitted",
        "negative_integrity_checks","seconds")}))
    if (not result["completed_within_window"] or result["isolated_capture_passed"]!=result["planned"]
            or result["independent_reference_passed"]!=result["planned"]):
        raise SystemExit(1)
