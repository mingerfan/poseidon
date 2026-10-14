"""Bounded manual acceptance of the opt-in candidate helper capability; no API."""
import argparse,json,os,shlex,subprocess,sys,time,tarfile
from pathlib import Path
import numpy as np
from benchmark_graph import digest,require,samples
from benchmark_math import evaluate
from benchmark_torch import evaluate as reference
from unified_chunk_cases import cases,source_for
from unified_graph_contract import prepare,validate_candidate
from compiler_configuration import configuration,PROFILE_SHA256
from hecate_python_env import ROOT,WORK,VENV
from benchmark_runner import dump
from semantic_benchmark_execution import runtime_sources
from audit_unified_candidate import verify_candidate


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=18)
    a=p.parse_args();require(1<=a.limit<=18,'Manual batch limit')
    require(os.environ.get('IN_NIX_SHELL') and Path(sys.prefix)==VENV,'Locked Nix Python required')
    out=a.output;require(out.resolve().is_relative_to(WORK/'results') and not out.exists(),'Preserve evidence')
    out.mkdir();sources=runtime_sources();source_hash=digest(sources);start=time.monotonic()
    selected=cases()[:a.limit]
    plan=dict(schema=1,source_sha256=source_hash,sources=sources,runner_sha256=__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
              case_ids=[r['name'] for r in selected],max_wall_seconds=1800,max_result_bytes=512*1024**2,
              native_concurrency=1,agent_calls=0,evaluation='manual_candidate_not_agent')
    dump(out/'plan.json',plan)
    with tarfile.open(out/'frozen-source.tar.gz','w:gz') as tar:
        for name in sources:tar.add(ROOT/name,arcname=name)
        tar.add(Path(__file__),arcname=str(Path(__file__).relative_to(ROOT)))
    records=[];rows=[]
    try:
        for row in selected:
            require(time.monotonic()-start<1800,'Batch wall budget')
            require(runtime_sources()==sources,'Source changed')
            size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+sum(r.get('evidence_bytes',0) for r in rows)
            require(size<=512*1024**2,'Batch evidence budget')
            path=out/row['name'];path.mkdir()
            graph=row['model'];dump(path/'model.json',graph)
            request=prepare(graph,PROFILE_SHA256,configuration('seal-cpu-eva-w45-v1'),
                chunk_period=row['period'],construction_profile=row.get('profile'),construction=row.get('exercise'))
            source=source_for(row,request);(path/'candidate.py').write_text(source)
            candidate=dict(schema=1,request_id=request['request_id'],hecate_source=source)
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
                     '--compiler-configuration','seal-cpu-eva-w45-v1','--unified-chunk-period',str(row['period'])]
            if row.get('profile'):command += ['--unified-profile','public-v1']
            if row.get('exercise'):command += ['--unified-exercise',row['exercise']]
            with (path/'run.log').open('w') as f:
                process=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=min(300,max(1,1800-(time.monotonic()-start))))
            text=(path/'run.log').read_text()
            matches=[line.split('Candidate evidence: ',1)[1] for line in text.splitlines() if line.startswith('Candidate evidence: ')]
            item=dict(id=row['name'],exit_code=process.returncode,reference_probes=16,reference_max=reference_max,
                      physical_inputs=len(request['layout']['inputs']),physical_outputs=request['layout']['output_ciphertexts'])
            if matches:
                folder=Path(matches[-1]);report=json.loads((folder/'report.json').read_text())
                item.update(evidence=str(folder),status=report['status'],evidence_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()))
                if report['status']=='passed':
                    record=verify_candidate(folder)
                    record['case_id']=row['name'];records.append(record)
            else:item['status']='runner_failed'
            rows.append(item);dump(out/'progress.json',dict(rows=rows,completed=len(rows)))
            print(row['name'],item['status'],flush=True)
        require(runtime_sources()==sources and __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()==plan['runner_sha256'],'Final integrity')
        total=sum(r['comparison']['compared_values'] for r in records)
        blocked=sum(row['status']=='blocked' for row in rows)
        summary=dict(planned=len(selected),passed=len(records),blocked=blocked,failed=len(rows)-len(records)-blocked,skipped=0,
            records=records,rows=rows,compared_values=total,
            max_absolute_error=max((r['comparison']['max_absolute_error'] for r in records),default=None),
            weighted_mae=sum(r['comparison']['mae']*r['comparison']['compared_values'] for r in records)/total if total else None,
            seconds=time.monotonic()-start,agent_calls=0,source_sha256=source_hash,
            evaluation='manual_candidate_not_agent',scope='bounded chunk layout fixtures; not full Agent or helper coverage')
        dump(out/'report.json',summary)
        return int(summary['failed']!=0)
    except Exception as error:
        dump(out/'failure.json',dict(error=str(error),rows=rows,seconds=time.monotonic()-start,agent_calls=0))
        raise

if __name__=='__main__':raise SystemExit(main())
