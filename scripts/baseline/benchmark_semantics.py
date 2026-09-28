"""Coverage requirements, not claims of observed execution."""
import ast
import json
from collections import Counter
from pathlib import Path
from benchmark_graph import digest, validate, require
from benchmark_suite import ROOT, HELPERS, BOOTSTRAP

BASE=ROOT/"scripts/baseline"
CONSTRUCTS={
    "native.call": "typed helper calls and nested/forward calls",
    "native.return": "scalar, multiple, Plain and structural empty returns",
    "array.storage": "Expr cells, indexing, zero-dimensional/matrix storage",
    "array.view_copy": "view aliasing versus independent copies",
    "array.reshape_transpose": "outer array transforms, not ciphertext rotation",
    "array.arithmetic": "outer storage broadcast and arithmetic",
    "array.mutation": "name-target mutation and alias-sensitive output",
    "scalar.augmented": "rebinding versus existing Expr aliases",
    "public.loop": "bounded construction-time range loops",
    "native.starred": "first-axis positional argument expansion",
}
COMPILER=("rescale","modswitch","relinearization","security_parameters","rotation_keys")
from rejection_benchmark import PARTITIONS as REJECT, tasks as rejection_tasks


def source_inventory():
    """Read AST/data only: importing upstream poly triggers I/O and unsupported dependencies."""
    sources={}
    api=[]
    relative=("third_party/dacapo/python/poly/poly/Func.py",
              "third_party/dacapo/python/poly/poly/MPCB.py",
              "third_party/dacapo/python/poly/poly/Poly.py",
              "third_party/dacapo/python/hecate/hecate/expr.py")
    import hashlib
    for path in relative:
        p=ROOT/path; raw=p.read_bytes(); sources[path]=hashlib.sha256(raw).hexdigest()
        tree=ast.parse(raw)
        def visit(body,prefix=""):
            for n in body:
                if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
                    q=prefix+n.name
                    api.append(dict(source=path,line=n.lineno,symbol=q,
                        kind="class" if isinstance(n,ast.ClassDef) else "function",
                        role=("public_helper" if n.name in HELPERS else
                              "dependency_or_frontend_api_requires_classification")))
                    visit(n.body,q+".")
        visit(tree.body)
    found={x["symbol"] for x in api if x["role"]=="public_helper"}
    if found!=set(HELPERS): raise ValueError("Upstream helper inventory drift")
    for path in (ROOT/"third_party/dacapo/python/poly/poly/data").glob("*"):
        if path.is_file(): sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return api,sources


def features(model):
    check=validate(model); values=check["shapes"]; result=set()
    for inp in model["inputs"]:
        result.add("input.rank."+str(len(inp["shape"])))
    import math
    maximum=max(math.prod(s["shape"]) for s in model["inputs"])
    bucket=next(n for n in (4,8,16,32,64,128,256) if n>=maximum)
    result.add("input.size_bucket."+str(bucket))
    result.add("input.count."+str(len(model["inputs"])))
    result.add("output.count."+str(len(model["outputs"])))
    consumers=Counter()
    for n in model["nodes"]:
        op=n["op"]; result.add("op."+op)
        for r in set(n["inputs"]):
            if r not in model["constants"]: consumers[r]+=1
        for k,v in n["attrs"].items():
            result.add("parameter."+op+"."+k+"."+json.dumps(v,separators=(",",":")))
        if op in ("add","subtract","multiply"):
            right=n["inputs"][1]
            if right in model["constants"]:
                s=values[right]
                result.add("overload."+op+".public."+("scalar" if not s else
                           "length1" if s==[1] else "tensor"))
            else: result.add("overload."+op+".cipher")
    if any(n>1 for n in consumers.values()): result.add("graph.shared_value")
    return sorted(result)


def ledger(rows):
    api,sources=source_inventory()
    coverage={}
    for row in rows:
        for f in features(row["model"]): coverage.setdefault(f,[]).append(row["model"]["id"])
    constructs={k:dict(instruction=v,source="scripts/baseline/native_function_rules.py")
                for k,v in CONSTRUCTS.items()}
    for path in sorted(BASE.glob("*exercises-v1.json")):
        data=json.loads(path.read_text())
        if type(data) is not dict: continue
        for name,entry in data.items():
            if type(entry) is not dict: continue
            spec=entry.get("spec",entry)
            if type(spec) is dict and spec.get("required_features"):
                for feature in spec["required_features"]:
                    constructs.setdefault(feature,dict(instruction=spec.get("instruction",""),
                        source=str(path.relative_to(ROOT)),legacy_exercise=name))
    # Native array/star catalogs are literal Python tables, read without import.
    for path in sorted(BASE.glob("*_exercises.py")):
        tree=ast.parse(path.read_bytes())
        for node in tree.body:
            if not isinstance(node,ast.Assign) or not any(isinstance(t,ast.Name) and t.id=="EXERCISES" for t in node.targets):
                continue
            try: table=ast.literal_eval(node.value)
            except (ValueError,TypeError): continue
            for name,entry in table.items():
                if isinstance(entry,(list,tuple)) and len(entry)==3:
                    _,required,instruction=entry
                    for feature in required:
                        constructs.setdefault(feature,dict(instruction=instruction,
                            source=str(path.relative_to(ROOT)),legacy_exercise=name))
    requirements=[]
    for f,ids in sorted(coverage.items()):
        requirements.append(dict(id=f,layer="model",models=ids,source="benchmark_graph.py",
                                 status="specified",evidence=[],blocker=None))
    # Even absent operators must remain explicit denominator entries.
    from benchmark_graph import OPS
    for op in OPS:
        if "op."+op not in coverage:
            requirements.append(dict(id="op."+op,layer="model",models=[],source="benchmark_graph.py",
                                     status="missing_fixture",evidence=[],blocker="fixture_gap"))
    from unified_graph_exercises import SPECS,STRUCTURAL,spec as native_spec
    from unified_public_exercises import SPECS as PUBLIC_SPECS,STRUCTURAL as PUBLIC_STRUCTURAL
    from unified_public_exercises import EVIDENCE_SCOPES,spec as public_spec
    from unified_public_exercises import VIEW_RECIPES,LAMBDA_RECIPES
    for feature in VIEW_RECIPES:
        constructs.setdefault(feature,dict(instruction='Observable object storage distinction',
                              source='scripts/baseline/unified_public_views.py'))
    for feature in LAMBDA_RECIPES:
        constructs.setdefault(feature,dict(instruction='Observable bound lambda parameter',
                              source='scripts/baseline/unified_public_interventions.py'))
    public_bound={v[0]:k for k,v in PUBLIC_SPECS.items()}
    bound={v[0]:k for k,v in SPECS.items()}
    if set(bound)&set(public_bound):raise ValueError("Conflicting directed profile bindings")
    bound.update(public_bound);specifications=dict(SPECS,**PUBLIC_SPECS)
    tasks=[]
    # Every construct gets distinct topology contexts. These are planned, not compatible-by-assertion.
    contexts=[]; seen=set()
    for row in sorted(rows,key=lambda r:(r["category"]!="graph",r["model"]["id"])):
        if row["topology"] not in seen:
            contexts.append(row);seen.add(row["topology"])
    for index,(name,spec) in enumerate(sorted(constructs.items())):
        ids=[]
        eligible=[r for r in contexts if name!="golden.zero" or len(r["model"]["outputs"])==1]
        if len(eligible)<3:raise ValueError("Insufficient independent contexts")
        for j in range(3):
            row=eligible[(index*3+j)%len(eligible)]; ids.append(row["model"]["id"])
            required=(public_spec(bound[name]) if name in public_bound else native_spec(bound[name]))["required_features"] if name in bound else [name]
            structural=[f for f in required if f in STRUCTURAL|PUBLIC_STRUCTURAL]
            acceptance="mixed" if structural and len(structural)<len(required) else "structure_only" if structural else "numeric_influence"
            task=dict(id="construct_"+str(index).zfill(3)+"_"+str(j),track="directed",
                      model_id=row["model"]["id"],model_sha256=row["model_sha256"],
                      requirement=name,instruction=specifications[bound[name]][1] if name in bound else spec["instruction"],
                      legacy_instruction=spec["instruction"],
                      acceptance_kind=acceptance,required_features=required,structural_features=structural,
                      evidence_scope=EVIDENCE_SCOPES.get(name,"executed_operation"),
                      profile=("hecate-unified-public-v1" if name in public_bound else "hecate-unified-native-v1") if name in bound else "unassigned_pending_compatibility",
                      exercise=bound.get(name),
                      checker_status="finite_intervention_and_real_trace" if name in bound else "not_bound",execution_status="not_run")
            task["task_sha256"]=digest(task);tasks.append(task)
        requirements.append(dict(id=name,layer="construction",models=ids,source=spec["source"],
                                 status="planned",evidence=[],blocker=None if name in bound else "contract_and_witness_binding"))
    from upstream_helper_directed_cases import tasks as helper_tasks
    directed_helpers=helper_tasks()
    for helper in HELPERS:
        ids=[r["model"]["id"] for r in rows if r["metadata"].get("helper")==helper]
        requirements.append(dict(id="helper."+helper,layer="upstream_helper",models=ids,
            source=next(x for x in api if x["symbol"]==helper),
            status="planned" if helper in ('HE_BN','HE_SiLU','HE_Concat','HE_Conv','HE_Avg','HE_Pool','HE_ConvBN','HE_DwConv','HE_DS','HE_MPBN','HE_Linear','HE_ReshapeLinear') else "blocked",
            evidence=[],blocker=None if helper in ('HE_BN','HE_SiLU','HE_Concat','HE_Conv','HE_Avg','HE_Pool','HE_ConvBN','HE_DwConv','HE_DS','HE_MPBN','HE_Linear','HE_ReshapeLinear') else "bootstrap_backend" if helper in BOOTSTRAP else "helper_adapter_unverified",
            directed_task_ids=[t['id'] for t in directed_helpers if any(n.split('_chunk',1)[0].rstrip('0123456789')==helper for n in t['required_helpers'])],
            capability_scope=('bound_batch1_rank2_to4_prefix_fits_period' if helper=='HE_BN' else 'fixed_polynomial_no_bootstrap' if helper=='HE_SiLU' else 'two_equal_shape_flat_or_batch1_channel_concat_fits_period' if helper=='HE_Concat' else 'bound_batch1_periodic_or_public_window_maps_within_work_budget_without_bootstrap' if helper in ('HE_Conv','HE_Avg','HE_Pool') else 'bound_conv2d_bn_edge_exact_public_mean_shift_and_maps_within_work_budget' if helper in ('HE_ConvBN','HE_DwConv') else 'bound_two_spatial_slices_step_two_with_actual_centering_and_public_maps' if helper=='HE_DS' else 'full_virtual_ring_fixed_prefix_or_explicit_chunked_bindings_within_work_budget' if helper in ('HE_MPBN','HE_Linear','HE_ReshapeLinear') else None)))
    for helper in ("HE_MPBN","HE_Linear","HE_ReshapeLinear"):
        matches=[t for t in directed_helpers if t.get("chunk_period") is not None
                 and any(n.split("_chunk",1)[0].rstrip("0123456789")==helper for n in t["required_helpers"])]
        require(len({t["topology"] for t in matches})>=3,"Chunk helper context coverage")
        requirements.append(dict(id="helper.chunked."+helper,layer="upstream_helper_layout",
            models=[t["model"]["id"] for t in matches],source="scripts/baseline/upstream_adapters/chunk_virtual_node.py",
            applicable_contract="upstream-poly-chunked-virtual-v9",status="planned",evidence=[],blocker=None,
            directed_task_ids=[t["id"] for t in matches],
            acceptance="actual_trace_and_finite_return_intervention_and_encrypted_numerics",
            scope="complete logical input assembled from independent chunks; immutable per-output chunk projection"))
    for name in COMPILER:
        requirements.append(dict(id="compiler."+name,layer="compiler",models=[],status="not_run",
            source="dsl_semantic_inventory.py",evidence=[],blocker="requires_real_artifact_evidence",
            evidence_protocol="hevm-runtime-evidence-v1",checker_source="scripts/baseline/compiler_artifact_evidence.py",
            evidence_scope="HEVM operations and observed final ciphertext metadata; not per-instruction runtime tracing or all-input proof"))
    negatives=rejection_tasks()
    for name in REJECT:
        matching=[t for t in negatives if t["requirement"]=="reject."+name]
        requirements.append(dict(id="reject."+name,layer="rejection",models=[],status="test_required",
            source="scripts/baseline/rejection_benchmark.py",evidence=[],blocker=None,
            task_ids=[t["id"] for t in matching],acceptance_layer=matching[0]["layer"],
            evidence_scope=matching[0]["scope"]))
    # Do not turn an API-name inventory into behavioral coverage.
    return dict(version="semantic-ledger-v1",requirements=requirements,upstream_api=api,
                upstream_source_hashes=sources,directed_tasks=tasks,helper_directed_tasks=directed_helpers,rejection_tasks=negatives,
                evidence_counts=dict(actual_helper_calls=0,encrypted_cases=0,agent_cases=0),
                full_semantics_proven=False)


def legacy_overlap(rows):
    """Compare real legacy graphs after normalization; never infer historical success."""
    from benchmark_bridge import from_legacy
    from benchmark_graph import signature
    p=BASE/"cases/user-graph-suite-v2.json"
    data=json.loads(p.read_text())
    old={}
    errors=[]
    for g in data["cases"]:
        try: old.setdefault(signature(from_legacy(g)),[]).append(g["id"])
        except ValueError as error: errors.append(dict(id=g["id"],reason=str(error)))
    return dict(legacy_cases=len(data["cases"]),normalization_errors=errors,
                overlaps=[dict(model_id=r["model"]["id"],legacy_ids=old[r["signature"]])
                          for r in rows if r["signature"] in old],
                inherits_historical_execution_evidence=False)
