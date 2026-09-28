"""48-case resource pilot, selected from public graph facts before FHE outcomes."""
import argparse,collections,hashlib,json,os,resource,signal,subprocess,time,tarfile
from pathlib import Path
from benchmark_runner import DEFAULT,ROOT,load,dump
from benchmark_graph import digest,require,validate
from semantic_benchmark_execution import runtime_sources,preflight,CONFIG
from hecate_python_env import WORK,VENV
from audit_unified_candidate import verify_candidate

def select(rows,items):
    ready=[x for x in items if x["status"]=="ready"];selected=[];reasons={}
    def add(item,why):
        reasons.setdefault(item["id"],[]).append(why)
        if item not in selected:selected.append(item)
    for category in sorted({x["category"] for x in ready}):
        group=sorted((x for x in ready if x["category"]==category),key=lambda x:(x["work"],x["id"]))
        for numerator,denominator,label in ((0,1,"minimum"),(1,2,"median"),(19,20,"p95"),(1,1,"maximum")):
            add(group[(len(group)-1)*numerator//denominator],category+":"+label+"_static_work")
    byid={r["model"]["id"]:r["model"] for r in rows}
    def size(item,output=False):
        g=byid[item["id"]];shapes=validate(g)["shapes"]
        import math
        return sum(math.prod(shapes[o["value"]]) for o in g["outputs"]) if output else sum(math.prod(i["shape"]) for i in g["inputs"])
    add(max(ready,key=lambda x:(size(x),x["work"],x["id"])),"maximum_total_input_elements")
    add(max(ready,key=lambda x:(size(x,True),x["work"],x["id"])),"maximum_total_output_elements")
    add(max(ready,key=lambda x:(len(byid[x["id"]]["nodes"]),x["work"],x["id"])),"maximum_node_count")
    prefixes=("op.","input.count.","output.count.","input.rank.","input.size_bucket.")
    core=lambda x:{f for f in x["features"] if f.startswith(prefixes)}
    covered=set().union(*(core(x) for x in selected))
    while len(selected)<48:
        candidates=[x for x in ready if x not in selected]
        x=max(candidates,key=lambda x:(len(core(x)-covered),-x["work"],x["id"]))
        add(x,"remaining_semantic_coverage");covered.update(core(x))
    require(len(selected)==len({x["id"] for x in selected})==48,"48 unique pilot models")
    return selected,reasons,sorted(covered)

def sample_tree(pid):
    table={}
    for f in Path("/proc").glob("[0-9]*/stat"):
        try:
            parts=f.read_text().rsplit(") ",1)[1].split()
            table[int(f.parent.name)]=(int(parts[1]),int(parts[21])*os.sysconf("SC_PAGE_SIZE")//1024)
        except (OSError,ValueError,IndexError):pass
    ids={pid};changed=True
    while changed:
        more={n for n,(parent,_) in table.items() if parent in ids}-ids
        changed=bool(more);ids.update(more)
    return sum(table[n][1] for n in ids if n in table),len(ids)

def available():
    fields=dict(line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].strip().split()[0])

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true");args=parser.parse_args()
    rows,index=load(DEFAULT);sources=runtime_sources();items=preflight(rows);selected,reasons,coverage=select(rows,items)
    body=dict(schema=1,source_sha256=digest(sources),sources=sources,index_sha256=hashlib.sha256((DEFAULT/"index.json").read_bytes()).hexdigest(),
       compiler_configuration=CONFIG,selection=[dict(x,reasons=reasons[x["id"]]) for x in selected],
       preflight=items,covered_features=coverage,native_concurrency=1,max_wall_seconds=1800,max_retained_bytes=512*1024**2,
       max_observed_filesystem_growth_bytes=2*1024**3,per_case_seconds=300,
       selection_uses_observed_FHE_results=False,paid_api_calls=0,
       runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    body["binding"]=digest(body)
    if not args.execute:
        print(json.dumps(dict(binding=body["binding"],selected=body["selection"],
          ready=sum(x["status"]=="ready" for x in items),blocked=sum(x["status"]!="ready" for x in items),
          selected_categories=dict(collections.Counter(x["category"] for x in selected)),features=coverage,paid_api_calls=0),indent=2))
        return 0
    out=WORK/"results/benchmark-r30-resource-pilot";require(not out.exists(),"Preserve pilot evidence");out.mkdir();dump(out/"plan.json",body)
    with tarfile.open(out/"source.tar.gz","w:gz") as archive:
        for name in sources:archive.add(ROOT/name,arcname=name)
        archive.add(Path(__file__),arcname=str(Path(__file__).relative_to(ROOT)))
    start=time.monotonic();free_start=os.statvfs(WORK).f_bavail*os.statvfs(WORK).f_frsize
    minimum_available=available();peak_growth=0;results=[];audit=[];byid={r["model"]["id"]:r for r in rows}
    for item in selected:
        require(runtime_sources()==sources,"Source drift")
        used=sum(x.get("evidence_bytes",0) for x in results)+sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        if time.monotonic()-start>=1800 or used>=body["max_retained_bytes"]:break
        name=item["id"];model=out/(name+".model.json");dump(model,byid[name]["model"])
        log=out/(name+".log");metrics=out/(name+".process-resources.json")
        argv=[str(ROOT/"scripts/baseline/run_candidate.py"),"--inside","--case",str(model),"--self-test","--max-repairs","0","--compiler-configuration",CONFIG]
        wrapper=("import runpy,sys,resource,json; sys.argv=sys.argv[1:]; sys.path.insert(0,"+repr(str(ROOT/"scripts/baseline"))+")\n"
                 "try: runpy.run_path(sys.argv[0],run_name='__main__')\n"
                 "finally:\n a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);"
                 "open("+repr(str(metrics))+",'w').write(json.dumps(dict(max_individual_process_rss_kib=max(a.ru_maxrss,b.ru_maxrss),"
                 "cpu_user_seconds=a.ru_utime+b.ru_utime,cpu_system_seconds=a.ru_stime+b.ru_stime)))")
        begin=time.monotonic();peak_tree=0;snapshots=0;reason=None
        with log.open("w") as f:
            child=subprocess.Popen([str(VENV/"bin/python"),"-B","-c",wrapper,*argv],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                rss,count=sample_tree(child.pid);peak_tree=max(peak_tree,rss);snapshots+=1
                minimum_available=min(minimum_available,available())
                stat=os.statvfs(WORK);growth=max(0,free_start-stat.f_bavail*stat.f_frsize);peak_growth=max(peak_growth,growth)
                if time.monotonic()-start>=1800:reason="phase_wall_budget"
                elif time.monotonic()-begin>=300:reason="case_wall_budget"
                elif growth>body["max_observed_filesystem_growth_bytes"]:reason="filesystem_growth_budget"
                if reason:
                    os.killpg(child.pid,signal.SIGINT)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
                    break
                time.sleep(.25)
        text=log.read_text();paths=[l.split("Candidate evidence: ",1)[1] for l in text.splitlines() if l.startswith("Candidate evidence: ")]
        result=dict(id=name,category=item["category"],work=item["work"],split=byid[name]["split"],seconds=time.monotonic()-begin,
            exit_code=child.returncode,status="harness_failed",resource_stop=reason,peak_sampled_tree_rss_kib=peak_tree,samples=snapshots,
            log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),model_sha256=byid[name]["model_sha256"])
        if metrics.exists():result.update(json.loads(metrics.read_text()))
        if paths:
            folder=Path(paths[-1]);report=json.loads((folder/"report.json").read_text())
            result.update(status=report["status"],evidence=str(folder),report_sha256=hashlib.sha256((folder/"report.json").read_bytes()).hexdigest(),
              evidence_bytes=sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()),
              attempts=[{k:a[k] for k in ("compiled","executed","numerically_correct","failure_layer","diagnostic") if k in a} for a in report["attempts"]])
            if report["status"]=="passed":audit.append(verify_candidate(folder))
        results.append(result);dump(out/"progress.json",dict(completed=len(results),rows=results));print(name,result["status"],flush=True)
        if reason or result["status"]=="harness_failed":break
    require(runtime_sources()==sources,"Final source drift")
    summary=dict(schema=1,binding=body["binding"],source_sha256=digest(sources),planned=48,completed=len(results),
        passed=sum(r["status"]=="passed" for r in results),failed=sum(r["status"]!="passed" for r in results),not_run=48-len(results),
        seconds=time.monotonic()-start,rows=results,records=audit,
        max_individual_process_rss_kib=max((r.get("max_individual_process_rss_kib",0) for r in results),default=0),
        peak_sampled_tree_rss_kib=max((r["peak_sampled_tree_rss_kib"] for r in results),default=0),
        memory_method="Individual kernel high-water mark; sampled sum of descendant RSS may double-count shared pages and miss short peaks",
        guest_minimum_mem_available_kib=minimum_available,peak_observed_filesystem_growth_bytes=peak_growth,
        retained_evidence_bytes=sum(r.get("evidence_bytes",0) for r in results),
        pilot_directory_bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
        source_changed=False,paid_api_calls=0,upstream_helper_execution_claimed=False)
    dump(out/"report.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("rows","records")},indent=2))
    return int(summary["not_run"]>0 or any(r["resource_stop"] or r["status"]=="harness_failed" for r in results))
if __name__=="__main__":raise SystemExit(main())
