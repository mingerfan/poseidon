"""Observe an existing native process; no new native execution is launched."""
import argparse
from datetime import datetime, timezone
import fcntl,json,os,select,shutil,signal,time
from pathlib import Path
from .. import paths
from .completion_metrics import artifact_bytes,persist_case_completion
from .r10_paper_metrics import StageRecorder,read_events,summarize_events,write_json
GIB=1024**3
ROUND='R31-SURVIVOR-OBSERVATION'
MODE='existing-native-process-after-supervisor-interruption'

def process_identity(pid):
 p=Path('/proc')/str(pid)
 try:
  raw=(p/'stat').read_text();tail=raw[raw.rindex(')')+2:].split()
  s={x.split(':',1)[0]:x.split(':',1)[1].strip() for x in (p/'status').read_text().splitlines() if ':' in x}
  return dict(pid=pid,state=tail[0],ppid=int(tail[1]),starttime_ticks=int(tail[19]),vmrss_kib=int(s.get('VmRSS','0 kB').split()[0]),vmhwm_kib=int(s.get('VmHWM','0 kB').split()[0]))
 except FileNotFoundError:return None

def identity_matches(o,pid,start):return o is not None and o['pid']==pid and o['starttime_ticks']==start

def available_ram():
 return int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024

def guard_reason(o,available,free,size,elapsed):
 if o and o['vmrss_kib']*1024>14*GIB:return 'native_RSS_above14GiB'
 if available<1.25*GIB:return 'available_RAM_below1.25GiB'
 if free<10*GIB:return 'free_disk_below10GiB'
 if size>4*GIB:return 'original_case_above4GiB'
 if elapsed>=10800:return 'original_native_wall_above10800seconds'
 return None

def terminal_status(client,alive,reason=None):
 if reason:return 'observer_stopped_native'
 if alive:return 'observing'
 if client.get('status')=='ok' and client.get('encrypted_plan_completed') is True:return 'native_completed_observed'
 if client.get('status')=='error':return 'native_error_observed'
 return 'native_exited_without_terminal_client'

def execute(a):
 out=a.out.resolve();source=a.source_case.resolve()
 if out.exists() or out.parent!=paths.RESULTS_DIR.resolve() or out==source:raise ValueError('Fresh observer directory required')
 starts=[x for x in read_events(source/'events.jsonl') if x.get('type')=='begin' and x.get('stage_id')=='execute']
 if len(starts)!=1:raise ValueError('One original execute start required')
 original=starts[0]['start_ns'];first=process_identity(a.pid)
 if not identity_matches(first,a.pid,a.starttime) or first['state']=='Z' or first['ppid']!=1:raise ValueError('Survivor identity mismatch')
 cmd=Path('/proc',str(a.pid),'cmdline').read_bytes().split(b'\0');exe=Path('/proc',str(a.pid),'exe').resolve()
 if exe.name!='r31-client' or str(source/'client.json').encode() not in cmd or b'execute' not in cmd:raise ValueError('Wrong native command')
 started=time.perf_counter_ns();out.mkdir()
 r=dict(round=ROUND,mode=MODE,model='resnet20_silu',status='observing',source_case=str(source),native_pid=a.pid,native_starttime_ticks=a.starttime,native_executable=str(exe),initial_identity=first,
 original_supervisor_exit_code=143,original_supervisor_signal=15,original_supervisor_cause='unknown',original_supervisor_clean=False,original_continuous_outer_completion=None,
 native_exit_code=None,waitpid_available=False,native_spawned=0,native_restarted=0,additional_encryption=0,additional_keygen=0,additional_CKKS_runs=0,llm_calls=0,
 original_execute_start_ns=original,observation_start_ns=started,observation_start_utc=datetime.now(timezone.utc).isoformat(),native_client=str(source/'client.json'),native_events=str(source/'go-events.jsonl'),
 original_resources=str(source/'execute-resources.jsonl'),resources=str(out/'resources.jsonl'),guard_stop=None,termination_signals_sent=[],error=None,security_acceptance=None,formal_latency=None,extra_integrity_checks=0)
 f=source/'execute-resources.jsonl'
 with f.open('rb') as stream:
  stream.seek(max(0,f.stat().st_size-10000));last=json.loads(stream.read().splitlines()[-1])
 r['last_original_resource_ns']=last['ts_ns'];r['resource_observation_gap_seconds']=(started-last['ts_ns'])/1e9
 shutil.copyfile(__file__,out/'observer-source.py');write_json(out/'result.json',r)
 pending=[];old={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
 def handler(s,frame):pending.append(s)
 for s in old:signal.signal(s,handler)
 fd=os.pidfd_open(a.pid);poll=select.poll();poll.register(fd,select.POLLIN);rec=StageRecorder(out/'events.jsonl');last_budget=0;size=artifact_bytes(source)
 def alive():
  o=process_identity(a.pid);return identity_matches(o,a.pid,a.starttime) and o['state']!='Z' and not poll.poll(0)
 def stop_native():
  if alive():os.kill(a.pid,signal.SIGTERM);r['termination_signals_sent'].append(15)
  until=time.monotonic()+10
  while time.monotonic()<until and not poll.poll(100):pass
  if alive():os.kill(a.pid,signal.SIGKILL);r['termination_signals_sent'].append(9)
 try:
  with rec.span('surviving-native-observation'),(out/'resources.jsonl').open('a',buffering=1) as samples:
   while alive():
    o=process_identity(a.pid);now=time.perf_counter_ns();available=available_ram();free=shutil.disk_usage(source).free
    if now-last_budget>10e9:size=artifact_bytes(source);last_budget=now
    sample=dict(ts_ns=now,observed=o,available_ram_bytes=available,free_disk_bytes=free);samples.write(json.dumps(sample)+'\n')
    why='observer_incoming_signal_'+str(pending[0]) if pending else guard_reason(o,available,free,size,(now-original)/1e9)
    if why:
     r['guard_stop']=dict(reason=why,observation=sample)
     try:write_json(out/'guard-stop.json',r['guard_stop'])
     finally:stop_native()
     break
    poll.poll(200)
  r['final_identity']=process_identity(a.pid);r['native_alive_after_observation']=bool(alive())
  client=json.loads((source/'client.json').read_text());r['native_client_terminal_status']=client.get('status')
  r['native_client_counts']={k:client.get(k) for k in ('compute_completed','bootstrap_completed','encryption_completed','final_decryption_completed','intermediate_decryptions')}
  r['native_client_comparison']=client.get('comparison');r['status']=terminal_status(client,r['native_alive_after_observation'],r['guard_stop'])
 except BaseException as e:
  r.update(status='observer_error',error=type(e).__name__+': '+str(e));stop_native()
 finally:
  os.close(fd)
  for s,h in old.items():signal.signal(s,h)
  r['observation_end_ns']=time.perf_counter_ns();r['observation_end_utc']=datetime.now(timezone.utc).isoformat();r['observed_total_since_original_execute_seconds']=(r['observation_end_ns']-original)/1e9
  r['timing']=summarize_events(read_events(out/'events.jsonl'));write_json(out/'result.json',r)
  row=dict(run_id=out.name,round=ROUND,model=r['model'],mode=MODE,git_rev=None,metrics={k:r.get(k) for k in ('status','native_pid','native_starttime_ticks','native_exit_code','native_client_counts','native_client_comparison','guard_stop','original_supervisor_clean','original_continuous_outer_completion')},artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
  with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
   fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
  persist_case_completion(out,round_label=ROUND,started_ns=started,status=r['status'],timing=r['timing'],required_artifact_bytes=artifact_bytes(out))
 return r

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);p.add_argument('--source-case',required=True,type=Path);p.add_argument('--pid',required=True,type=int);p.add_argument('--starttime',required=True,type=int)
 r=execute(p.parse_args(argv));print(json.dumps({'status':r['status'],'native_exit_code':None}));return 0 if r['status']=='native_completed_observed' else 1
if __name__=='__main__':raise SystemExit(main())
