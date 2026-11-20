"""Persist an explicit50-bit native plan projection, without keys or crypto."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import shutil
import time

from .. import paths
from .r10_paper_metrics import StageRecorder,read_events,summarize_events,write_json
from .r31_lattigo_binding import NativeBinding,map_precision_plan
from .r31_lattigo_runtime_plan import plan_inventory
from .completion_metrics import artifact_bytes,persist_case_completion


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-plan','bundle','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--precision-bits',type=int,choices=(50,),default=50)
    p.add_argument('--s2c-bits',type=int,choices=(50,))
    p.add_argument('--mod1-inv-degree',type=int,choices=(7,15))
    p.add_argument('--mod1-degree',type=int,choices=(63,))
    p.add_argument('--log-message-ratio',type=int,choices=(2,))
    p.add_argument('--bootstrap-input-divisor',type=int,choices=(2,))
    p.add_argument('--bootstrap-normalization-log2',type=int,choices=(0,1,2,3),default=0)
    p.add_argument('--bootstrap-cache',type=int,choices=(1,16,32,40,44,48),default=48)
    args=p.parse_args(argv);out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter_ns();rec=StageRecorder(out/'events.jsonl');binding=NativeBinding(args.precision_bits,args.s2c_bits,args.mod1_inv_degree,args.mod1_degree,args.log_message_ratio,args.bootstrap_input_divisor,args.bootstrap_normalization_log2)
    r=dict(round='R31',mode='explicit-native-precision-projection',model='resnet20_silu',status='running',error=None,
        variant=binding.variant(args.bootstrap_cache),source_plan=str(args.source_plan),source_bundle=str(args.bundle),
        original_plan_modified=False,source_raw_data_modified=False,llm_calls=0,keygen_runs=0,real_ckks_runs=0,
        extra_integrity_checks=0,security_acceptance=None,accuracy_acceptance=None,artifact_bytes=0)
    try:
        for name in ('r31_lattigo_binding.py','r31_lattigo_precision.py'):
            shutil.copyfile(Path(__file__).with_name(name),out/name)
        with rec.span('source_plan_and_manifest_metadata'):
            original=json.loads(args.source_plan.read_text());manifest=json.loads((args.bundle/'manifest.json').read_text())
            r['source_inventory']=plan_inventory(original,manifest)
        with rec.span('explicit_precision_projection'):
            mapped,report=map_precision_plan(original,binding,bootstrap_cache=args.bootstrap_cache)
            r['mapped_inventory']=plan_inventory(mapped,manifest,binding=binding)
            if r['source_inventory']['op_counts']!=r['mapped_inventory']['op_counts']:
                raise ValueError('Native precision mapping changed original operations')
            r['precision_mapping']=report;r['derived_plan']=str(out/'derived-plan.json')
            write_json(out/'derived-plan.json',mapped)
        r['status']='ok'
    except BaseException as exc:r.update(status='error',error=type(exc).__name__+': '+str(exc))
    r['timing']=summarize_events(read_events(out/'events.jsonl'));r['elapsed_ns']=time.perf_counter_ns()-started
    r['metrics']={k:r[k] for k in ('status','variant','llm_calls','keygen_runs','real_ckks_runs','security_acceptance','accuracy_acceptance','elapsed_ns')}
    for _ in range(16):
        write_json(out/'result.json',r);size=artifact_bytes(out)
        if size==r['artifact_bytes']:break
        r['artifact_bytes']=size
    from ..registry import git_rev
    row=dict(run_id=out.name,round='R31',model=r['model'],mode=r['mode'],git_rev=git_rev(),metrics=r['metrics'],artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
    persist_case_completion(out,round_label='R31',started_ns=started,status=r['status'],timing=r['timing'],required_artifact_bytes=r['artifact_bytes'])
    print(json.dumps(dict(status=r['status'],error=r['error'],out=str(out))))
    return 0 if r['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
