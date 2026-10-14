"""Bounded manual acceptance of the opt-in candidate helper capability; no API."""
import argparse,json,os,shlex,subprocess,sys,time,tarfile,hashlib
from pathlib import Path
import numpy as np
from benchmark_graph import digest,require,samples
from benchmark_math import evaluate
from benchmark_torch import evaluate as reference
from upstream_helper_directed_cases import cases
from unified_graph_contract import prepare,validate_candidate
from upstream_candidate_helpers import BN_PROFILE as PROFILE,verify_sources
from poly_dependencies import verify
from compiler_configuration import configuration,PROFILE_SHA256
from hecate_python_env import ROOT,WORK,VENV
from benchmark_runner import dump
from semantic_benchmark_execution import runtime_sources
from audit_unified_candidate import verify_candidate


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=48)
    p.add_argument('--task-id',action='append');p.add_argument('--suite',type=Path);p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv);require(1<=a.limit<=48,'Manual batch limit')
    require(os.environ.get('IN_NIX_SHELL') and Path(sys.prefix)==VENV,'Locked Nix Python required')
    out=a.output;require(out.resolve().is_relative_to(WORK/'results') and (not out.exists() or a.resume),'Preserve evidence')
    require(not out.is_symlink(),'Indirect evidence directory')
    out.mkdir(exist_ok=a.resume);sources=runtime_sources();source_hash=digest(sources);start=time.monotonic()
    from benchmark_runner import load,DEFAULT,strict_file
    from upstream_helper_directed_cases import tasks
    suite=a.suite or DEFAULT;load(suite)
    frozen=strict_file(suite/'coverage.json',4*1024**2)['helper_directed_tasks']
    require(frozen==tasks(),'Directed helper task binding')
    wanted=a.task_id or [t['id'] for t in frozen]
    require(len(wanted)==len(set(wanted)) and set(wanted)<={t['id'] for t in frozen},'Unknown or repeated helper task')
    selected=[r for r in cases() if 'upstream_'+r['name'] in wanted][:a.limit]
    bound_tasks=[t for t in frozen if t['id'] in {'upstream_'+r['name'] for r in selected}]
    plan=dict(schema=1,source_sha256=source_hash,sources=sources,helper_sources=verify_sources(),dependency=verify(),
              case_ids=[r['name'] for r in selected],tasks=bound_tasks,suite=str(suite),max_wall_seconds=1800,max_result_bytes=512*1024**2,
              runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              native_concurrency=1,agent_calls=0,evaluation='manual_candidate_not_agent')
    previous=[];prior_seconds=0.
    if a.resume:
        require(json.loads((out/'plan.json').read_text())==plan,'Resume task/source/environment changed')
        if (out/'progress.json').exists():
            progress=json.loads((out/'progress.json').read_text());previous=progress['rows'];prior_seconds=progress['seconds']
            require(type(prior_seconds) in (int,float) and 0<=prior_seconds<=1800,'Resume elapsed budget')
        require(len({r['id'] for r in previous})==len(previous) and {r['id'] for r in previous}<=set(plan['case_ids']),'Resume row identity')
    else:dump(out/'plan.json',plan)
    if not a.resume:
      with tarfile.open(out/'frozen-source.tar.gz','w:gz') as tar:
        for name in sources:tar.add(ROOT/name,arcname=name)
        tar.add(Path(__file__),arcname=str(Path(__file__).relative_to(ROOT)))
    records=[];rows=[]
    try:
        for row in selected:
            require(prior_seconds+time.monotonic()-start<1800,'Batch wall budget')
            require(runtime_sources()==sources,'Source changed')
            size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+sum(r.get('evidence_bytes',0) for r in rows)
            require(size<=512*1024**2,'Batch evidence budget')
            done=next((r for r in previous if r['id']==row['name']),None)
            if done is not None:
                require(done['task_sha256']==next(t['task_sha256'] for t in bound_tasks if t['id']=='upstream_'+row['name']),'Resume task hash')
                if done['status']=='passed':
                    folder=Path(done['evidence'])
                    expected_request=prepare(row['model'],PROFILE_SHA256,configuration(row['configuration']),helper_profile=row.get('profile',PROFILE),helper_exercise=row['required_helpers'],chunk_period=row.get('chunk_period'))
                    require(json.loads((folder/'request.json').read_text())==expected_request and
                            (folder/'attempt-00/candidate.py').read_text()==row['source'],'Resume candidate/model identity')
                    record=verify_candidate(folder);record['case_id']=row['name'];records.append(record)
                rows.append(done);continue
            path=out/row['name']
            if path.exists():
                require(a.resume,'Preserve unfinished case')
                parent=path;index=0
                while (parent/('resume-'+str(index))).exists():index+=1
                path=parent/('resume-'+str(index))
            path.mkdir()
            graph=row['model'];dump(path/'model.json',graph);(path/'candidate.py').write_text(row['source'])
            request=prepare(graph,PROFILE_SHA256,configuration(row['configuration']),helper_profile=row.get('profile',PROFILE),helper_exercise=row['required_helpers'],chunk_period=row.get('chunk_period'))
            candidate=dict(schema=1,request_id=request['request_id'],hecate_source=row['source'])
            validate_candidate(candidate,request)
            reference_max=0.
            for probe in samples(graph,16):
                x=evaluate(graph,probe);y=reference(graph,probe)
                for name in x:
                    np.testing.assert_allclose(x[name],y[name],atol=1e-12,rtol=1e-12)
                    reference_max=max(reference_max,float(np.max(np.abs(x[name]-y[name]))))
            dump(path/'responses.json',[json.dumps(candidate)])
            command=[str(VENV/'bin/python'),'-B',str(ROOT/'scripts/baseline/run_candidate.py'),'--inside',
                     '--case',str(path/'model.json'),'--replay',str(path/'responses.json'),'--max-repairs','0',
                     '--compiler-configuration',row['configuration'],'--unified-helpers',row.get('profile',PROFILE)]
            if 'chunk_period' in row:command += ['--unified-chunk-period',str(row['chunk_period'])]
            for name in row['required_helpers']:command += ['--unified-helper-exercise',name]
            with (path/'run.log').open('w') as f:
                process=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=min(300,max(1,1800-(prior_seconds+time.monotonic()-start))))
            text=(path/'run.log').read_text()
            matches=[line.split('Candidate evidence: ',1)[1] for line in text.splitlines() if line.startswith('Candidate evidence: ')]
            item=dict(id=row['name'],task_sha256=next(t['task_sha256'] for t in bound_tasks if t['id']=='upstream_'+row['name']),exit_code=process.returncode,reference_probes=16,reference_max=reference_max,
                      expected_calls=row.get('expected_calls',1),contribution_negative=row.get('contribution_negative',False))
            if matches:
                folder=Path(matches[-1]);report=json.loads((folder/'report.json').read_text())
                item.update(evidence=str(folder),status=report['status'],evidence_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()))
                if report['status']=='passed':
                    record=verify_candidate(folder)
                    require(record['upstream_helper_trace']['actual_upstream_calls']==row.get('expected_calls',1),'Expected helper invocation count')
                    require(record['helper_coverage']['finite_return_influence_checked'] and record['helper_coverage']['actual_frontend_checked'],'Missing directed helper witness')
                    record['case_id']=row['name'];records.append(record)
            else:item['status']='runner_failed'
            rows.append(item);dump(out/'progress.json',dict(rows=rows,completed=len(rows),seconds=prior_seconds+time.monotonic()-start))
            print(row['name'],item['status'],flush=True)
        require(runtime_sources()==sources and verify_sources()==plan['helper_sources'] and verify()==plan['dependency'],'Final integrity')
        total=sum(r['comparison']['compared_values'] for r in records)
        summary=dict(planned=len(selected),passed=len(records),failed=len(rows)-len(records),skipped=0,
            records=records,rows=rows,compared_values=total,
            max_absolute_error=max((r['comparison']['max_absolute_error'] for r in records),default=None),
            weighted_mae=sum(r['comparison']['mae']*r['comparison']['compared_values'] for r in records)/total if total else None,
            seconds=prior_seconds+time.monotonic()-start,agent_calls=0,source_sha256=source_hash,
            directed_helper_contribution_proven=bool(records) and len(records)==len(selected),all_helpers_supported=False)
        if (out/'report.json').exists():
            index=0
            while (out/('report-before-resume-'+str(index)+'.json')).exists():index+=1
            (out/'report.json').rename(out/('report-before-resume-'+str(index)+'.json'))
        dump(out/'report.json',summary)
        return int(len(records)!=len(selected))
    except Exception as error:
        index=0
        while (out/('failure-'+str(index)+'.json')).exists():index+=1
        dump(out/('failure-'+str(index)+'.json'),dict(error=str(error),rows=rows,seconds=prior_seconds+time.monotonic()-start,agent_calls=0))
        raise

if __name__=='__main__':raise SystemExit(main())
