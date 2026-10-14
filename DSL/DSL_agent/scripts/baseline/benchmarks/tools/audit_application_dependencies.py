"""Read-only developer audit of actual stage source closures; no runtime or API."""
import argparse,collections,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:sys.path.insert(0,str(BASE))
from application_identity import source_catalog,_catalog_values,source_closure,_read,_digest,POLICY,CATALOG,STAGES

TARGETS=("application_component.py","deepseek_provider.py","run_candidate.py",
         "unified_graph_lowering.py","capacity_output_identity.py","trusted_pytorch_import.py")

def audit(root):
    root=Path(root);base=root/"scripts/baseline"
    catalog=source_catalog(base);roots,optional,authorities=_catalog_values(catalog)
    raw=_read(base,CATALOG);stages={}
    for stage in STAGES:
        selected=list(roots[stage])+["application_identity.py","application_backend.py"]
        selected.extend(name for name in optional.get(stage,()) if (base/name).is_file())
        closure=source_closure(base,selected,dynamic_authorities=authorities)
        queue=collections.deque((r,[r]) for r in closure["roots"]);seen=set();witnesses={}
        while queue:
            name,path=queue.popleft()
            if name in seen:continue
            seen.add(name)
            if name in TARGETS:witnesses[name]=path
            queue.extend((child,path+[child]) for child in closure["import_edges"].get(name,()))
        stages[stage]=dict(source_files=len(closure["sources"]),roots=closure["roots"],
            closure_sha256=_digest(closure),source_hashes=closure["sources"],target_paths=witnesses)
    if raw!=_read(base,CATALOG):raise ValueError("Identity catalog changed during audit")
    sets=[set(v["source_hashes"]) for v in stages.values()]
    shared=set.intersection(*sets)
    return dict(format="poseidon-application-dependency-audit-v1",policy=POLICY,
        catalog_sha256=__import__("hashlib").sha256(raw).hexdigest(),stages=stages,
        shared_source_files=len(shared),identical_source_sets=all(v==sets[0] for v in sets),
        runtime_execution=False,paid_calls=0,scope="actual source dependency discovery; not stage independence or encrypted validation")

def main(argv=None):
    parser=argparse.ArgumentParser(description="Inspect application stage source dependencies.")
    parser.add_argument("--root",type=Path,default=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4]));parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv);value=audit(args.root)
    if args.output:
        if args.output.exists():raise ValueError("Preserve existing dependency report")
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2)+"\n")
    print(json.dumps({k:v for k,v in value.items() if k!="stages"}))
    for stage,record in value["stages"].items():
        print(json.dumps(dict(stage=stage,source_files=record["source_files"],target_paths=record["target_paths"])))
    return 0
if __name__=="__main__":raise SystemExit(main())
