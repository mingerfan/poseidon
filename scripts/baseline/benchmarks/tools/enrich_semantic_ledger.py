"""Freeze a metadata-complete ledger without altering existing tasks or execution code.

Default is a read-only plan. --write creates a new explicit suite; it never
changes the default suite, imports old pass states, or mutates a running release.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; BASE=HERE.parents[1]; ROOT=BASE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import load,strict_file,dump
from verify_model_semantic_contexts import checked_bundle
from build_model_partition_rejections import build as build_rejections
from classify_upstream_semantic_api import build as classify
from verify_construction_trace_counterexamples import fixtures
from verify_helper_call_counterexamples import counterexamples
from semantic_benchmark_execution import runtime_sources

COMPONENTS={
 "math_contexts":"model-contexts-draft-v1/tasks.json",
 "model_negatives":"model-rejections-draft-v1/tasks.json",
 "api_classification":"upstream-api-classification-draft-v1/classification.json",
 "compiler_contexts":"compiler-contexts-draft-v2/examples.json",
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def grouped(rows,key):
    result={}
    for r in rows:result.setdefault(r[key],[]).append(r)
    return result
def source_basis(source):
    if isinstance(source,dict):
        path=source["source"];location={k:source[k] for k in ("line","symbol") if k in source}
    else:
        path=source if "/" in source else "scripts/baseline/"+source;location={}
    p=ROOT/path
    if not p.is_file() or p.is_symlink():raise ValueError("Unbound requirement source")
    return dict(path=path,sha256=sha(p),**location)
def ref(component,item,**extra):
    return dict(component=component,id=item,**extra)

def build(suite):
    rows,index=load(suite);old=strict_file(suite/"coverage.json",4*1024**2)
    paths={k:BASE/"benchmarks"/v for k,v in COMPONENTS.items()}
    components={k:strict_file(p,4*1024**2) for k,p in paths.items()}
    contexts,_=checked_bundle(paths["math_contexts"],suite)
    neg=components["model_negatives"]
    if neg["tasks"]!=build_rejections(rows,old):raise ValueError("Model rejection fixture drift")
    api=classify(old)
    if any(components["api_classification"].get(k)!=v for k,v in api.items()):
        raise ValueError("API classification drift")
    compiler=components["compiler_contexts"]
    if compiler["runtime_source_sha256"]!=digest(runtime_sources()):raise ValueError("Compiler evidence runtime drift")
    # This component links historical evidence. Its archive binding is checked here;
    # actual artifact re-audit remains bind_compiler_context_examples.py's job.
    if compiler["binding_sha256"]!=digest({k:v for k,v in compiler.items() if k!="binding_sha256"}):
        raise ValueError("Compiler component binding")
    binder=HERE/"bind_compiler_context_examples.py"
    if compiler["runner_sha256"]!=sha(binder):raise ValueError("Compiler binder drift")
    traces=fixtures(old);omissions=counterexamples(old["helper_directed_tasks"])
    components["trace_negatives"]=dict(schema=1,fixtures=traces,
        actual_frontend_executed=False,encrypted_execution=False)
    components["helper_negatives"]=dict(schema=1,fixtures=[
        dict(id=d["id"],positive_task_id=d["task"]["id"],positive_task_sha256=d["task"]["task_sha256"],
             omitted_helper=d["target"],expected_rejection="No reachable upstream helper: "+d["target"],
             status="not_run",runner="scripts/baseline/benchmarks/tools/verify_helper_call_counterexamples.py")
        for d in omissions],actual_frontend_executed=False,encrypted_execution=False)
    math_by=grouped(contexts["tasks"],"requirement")
    model_neg=grouped(neg["tasks"],"requirement");trace_by=grouped(traces,"requirement")
    directed=grouped(old["directed_tasks"],"requirement")
    reject=grouped(old["rejection_tasks"],"requirement")
    helpers={t["id"]:t for t in old["helper_directed_tasks"]}
    models={r["model"]["id"]:r for r in rows}
    result=copy.deepcopy(old)
    result["version"]="semantic-ledger-v2"
    result["upstream_api"]=components["api_classification"]["records"]
    result["semantic_protocol"]=dict(version="bounded-semantic-partitions-v2",
        coverage_claim="Only explicitly listed partitions, never all possible programs or all-input equivalence",
        canonical_requirements="requirements",original_model_count=1200,
        supplemental_model_count=len(contexts["supplemental_models"]),
        supplemental_models_counted_in_original=False,old_execution_states_imported=False,
        math_task_component="math_contexts",math_task_count=len(contexts["tasks"]),
        helper_negative_count=len(omissions),source_runtime_sha256=digest(runtime_sources()),
        parent_index_sha256=sha(suite/"index.json"),parent_coverage_sha256=index["coverage_sha256"])
    for r in result["requirements"]:
        key=r["id"];layer=r["layer"]
        r["source_basis"]=[source_basis(r["source"])]
        positive=[];negative=[];contracts=[];stages=[];scope=""
        if layer=="model":
            ts=math_by[key]
            positive=[ref("math_contexts",t["id"],model_id=t["model_id"],topology=t["topology"],
                          task_sha256=t["task_sha256"]) for t in ts]
            negative=[ref("model_negatives",t["id"],task_sha256=t["task_sha256"],
                          acceptance_layer="model_graph_validation") for t in model_neg[key]]
            contracts=["poseidon-model-graph-v1"]
            stages=["independent_dual_reference","actual_frontend","compiler_artifacts","SEAL_decrypt_numerical"]
            scope="Mathematical graph semantics; no particular DSL spelling is implied."
        elif layer=="construction":
            ts=directed[key];contracts=sorted({t["profile"] for t in ts})
            positive=[ref("directed_tasks",t["id"],task_sha256=t["task_sha256"],
                          topology=models[t["model_id"]]["topology"],acceptance_kind=t["acceptance_kind"]) for t in ts]
            negative=[ref("trace_negatives",t["id"],fixture_sha256=t["fixture_sha256"],
                          acceptance_layer="synthetic_trace_evidence_gate") for t in trace_by[key]]
            stages=["candidate_contract","actual_frontend_trace","finite_return_dependency_intervention",
                    "compiler_artifacts","SEAL_decrypt_numerical"]
            scope="Structural children score structure only; numeric children require their own output contribution. Synthetic trace negatives are checker tests."
        elif layer in ("upstream_helper","upstream_helper_layout"):
            ts=[helpers[n] for n in r["directed_task_ids"]]
            contracts=sorted({t["profile"] for t in ts})
            positive=[ref("helper_directed_tasks",t["id"],task_sha256=t["task_sha256"],
                          topology=t["topology"]) for t in ts]
            negative=[ref("helper_negatives",d["id"],acceptance_layer="candidate_AST_and_finite_probe")
                      for d in omissions if d["task"]["id"] in r["directed_task_ids"]]
            stages=["candidate_contract","actual_upstream_helper_trace","finite_return_dependency_intervention",
                    "compiler_artifacts","SEAL_decrypt_numerical"]
            scope=r.get("capability_scope") or r.get("scope") or "Backend-blocked real bootstrap; mathematical references are not helper execution."
            if r["blocker"]:
                contracts=["blocked_real_bootstrap"]
                positive=[ref("models",n,model_sha256=models[n]["model_sha256"],
                              topology=models[n]["topology"],state="blocked_helper_execution") for n in r["models"]]
                negative=[dict(kind="backend_blocker",reason=r["blocker"],state="not_an_executable_negative")]
        elif layer=="compiler":
            contracts=["hevm-runtime-evidence-v1"];stages=["artifact_lineage","actual_parameters_and_keys",
                "runtime_source_binding","SEAL_decrypt_numerical"]
            positive=[ref("compiler_contexts",key+":"+str(i),topology=e["topology"],
                          report_sha256=e["report_sha256"],observation="previously_verified_execution")
                      for i,e in enumerate(compiler["examples"][key])]
            # A shared real negative checks all partitions' prerequisite comparison identity.
            negative=[dict(kind="auditor_regression",
                source=source_basis("scripts/baseline/benchmarks/tools/test_compiler_context_bindings.py"),
                test="CompilerContextBindingTests.test_comparison_cannot_be_rewritten_with_still_passing_values",
                acceptance_layer="artifact_evidence_consistency",
                scope="Shared comparison-binding prerequisite; not a rejected compiler program")]
            scope=compiler["scope"]
        elif layer=="rejection":
            ts=reject[key];contracts=sorted({t["layer"] for t in ts})
            positive=[ref("rejection_tasks",t["id"],positive_sha256=t["positive_sha256"]) for t in ts]
            negative=[ref("rejection_tasks",t["id"],negative_sha256=t["negative_sha256"],
                          acceptance_layer=t["layer"]) for t in ts]
            stages=["specified_gate_positive_control","specified_gate_expected_rejection"];scope=ts[0]["scope"]
        else:raise ValueError("Unreviewed semantic layer")
        if layer!="rejection" and not r["blocker"] and len({x["topology"] for x in positive})<3:
            raise ValueError("Missing three distinct contexts: "+key)
        if not positive or not negative or not contracts or not stages:raise ValueError("Incomplete partition: "+key)
        r.update(applicable_contracts=contracts,positive_examples=positive,negative_examples=negative,
                 acceptance_levels=stages,claim_scope=scope,
                 blockers=[] if not r["blocker"] else [r["blocker"]])
    result["components"]={k:dict(file="semantic-component-"+k+".json",
        sha256=hashlib.sha256((json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()).hexdigest())
        for k,v in components.items()}
    result["metadata_generator"]=dict(path=str(Path(__file__).resolve().relative_to(ROOT)),sha256=sha(Path(__file__)))
    validate_enrichment(result,old)
    return rows,index,result,components

def validate_enrichment(result,old):
    for key in ("directed_tasks","helper_directed_tasks","rejection_tasks","evidence_counts"):
        if result[key]!=old[key]:raise ValueError("Original tasks or states changed: "+key)
    if len(result["requirements"])!=len(old["requirements"]):raise ValueError("Requirement denominator changed")
    if {r["id"] for r in result["requirements"]}!={r["id"] for r in old["requirements"]}:
        raise ValueError("Requirement identities changed")
    if result["full_semantics_proven"] is not False:raise ValueError("All-input proof overclaim")
    for r in result["requirements"]:
        if not all(r.get(k) for k in ("source_basis","applicable_contracts","positive_examples",
                                      "negative_examples","acceptance_levels","claim_scope")):
            raise ValueError("Missing required semantic metadata")
    raw=json.dumps(result,indent=2,sort_keys=True,allow_nan=False).encode()
    if len(raw)>4*1024**2:raise ValueError("Existing coverage file budget exceeded")

def verify_release(directory):
    """Validate the additional component closure; the legacy runner ignores metadata."""
    rows,index=load(directory);ledger=strict_file(directory/"coverage.json",4*1024**2)
    if ledger["version"]!="semantic-ledger-v2":raise ValueError("Wrong ledger version")
    if ledger["metadata_generator"]!=dict(path=str(Path(__file__).resolve().relative_to(ROOT)),sha256=sha(Path(__file__))):
        raise ValueError("Metadata generator drift")
    for name,entry in ledger["components"].items():
        if entry["file"]!="semantic-component-"+name+".json":raise ValueError("Unsafe component path")
        path=directory/entry["file"]
        strict_file(path,4*1024**2)
        if sha(path)!=entry["sha256"]:raise ValueError("Component integrity")
    for req in ledger["requirements"]:
        for source in req["source_basis"]:
            path=Path(source["path"])
            if path.is_absolute() or ".." in path.parts:raise ValueError("Unsafe source path")
            if sha(ROOT/path)!=source["sha256"]:raise ValueError("Requirement source drift")
    return dict(models=len(rows),requirements=len(ledger["requirements"]),components=len(ledger["components"]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",type=Path,required=True)
    p.add_argument("--output",type=Path);p.add_argument("--write",action="store_true");a=p.parse_args()
    rows,index,ledger,components=build(a.suite)
    summary=dict(requirements=len(ledger["requirements"]),models=len(rows),
        supplemental_models=ledger["semantic_protocol"]["supplemental_model_count"],
        api_records=len(ledger["upstream_api"]),directed_tasks=len(ledger["directed_tasks"]),
        helper_tasks=len(ledger["helper_directed_tasks"]),
        helper_negatives=ledger["semantic_protocol"]["helper_negative_count"],
        execution_states_imported=False,actual_new_execution=False,paid_calls=0)
    if a.write:
        if a.output is None or a.output.exists():p.error("New output directory required")
        if not a.output.resolve().is_relative_to((BASE/"benchmarks").resolve()):
            p.error("Freeze inside benchmark source directory")
        a.output.mkdir(parents=True)
        for entry in index["shards"]:
            (a.output/entry["file"]).write_bytes((a.suite/entry["file"]).read_bytes())
        (a.output/"legacy-overlap.json").write_bytes((a.suite/"legacy-overlap.json").read_bytes())
        for k,v in components.items():dump(a.output/ledger["components"][k]["file"],v)
        dump(a.output/"coverage.json",ledger)
        updated=copy.deepcopy(index);updated["coverage_sha256"]=sha(a.output/"coverage.json")
        dump(a.output/"index.json",updated)
        verify_release(a.output)
        summary.update(written=str(a.output),index_sha256=sha(a.output/"index.json"))
    print(json.dumps(summary))
    return 0
if __name__=="__main__":raise SystemExit(main())
