#!/usr/bin/env python3
"""LOCAL node2 aggregate-only parser, adapted from the read-only 20261115 reference.
Never connects, runs an app, or reads CT/PT. Raw captures are never changed.
The fixed complex-norm numerical gates are unchanged; partial refreshes are also checked.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,pathlib,re
HERE=pathlib.Path(__file__).resolve().parent
OUTPUTS=('paired_lane0','paired_lane1','old_lane0','old_lane1')
COND=('post_Gate_refresh_decodedUp_conditioned','post_all_refresh_decodedUp_conditioned')
THRESHOLD=0.001
ROOT='/root/private_data/llama-manual-framework-node2-20261116-v1'
EXPECTED_UUID='GPU-499236d3-b46c-5d25-4a22-530d47ed5112'
AUTH_SHA='ea218a2d67a0e04f5f439286b3784a2c915a7bbbb0e6682bbd68549929bb0d13'
GUARD_PID=95575
GUARD_START='1373031093'
EXPECTED_COMMAND=[ROOT+'/build/bin/llama31_public_paired_silu','--public-silu-pair-v1',ROOT+'/assets/r20_fixture.txt',ROOT+'/manifests/public-silu-baseline-v1.tsv',ROOT+'/reports/public-silu-baseline-v1/SUMMARY.json']
EXPECTED_ENV={'OMP_NUM_THREADS':'6','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','POSEIDON_GPU_QP_PREPARE':'optimized','POSEIDON_GPU_QP_PREPARE_THREADS':'4','POSEIDON_GPU_QP_VERIFY_REFERENCE':'0'}
CPU_MARKER='PUBLIC_SILU_PAIR_CPU_DONE cases=2 planned_physical_Boots=32 planned_projection_pairs=16 fixed_gate25_up8=true no_Runtime_keys_GPU=true'
COMMON_MARKER='PUBLIC_PROJECTIONS_CPU_DONE cases=10 physical_Boots=40 N=65536 slots=32768 pattern_zero_swap_sign_amplitude=true metadata8_word_mutations=true exclusive_report_symlink=true fixed_full_and_sine_gates=true active38_complex_norm=true no_Runtime_keys_GPU=true'

def strict_json(text):
 def pairs(seq):
  d={}
  for k,v in seq:
   if k in d:raise ValueError('duplicate key '+k)
   d[k]=v
  return d
 def bad(x):raise ValueError('nonfinite JSON '+x)
 def real(x):
  v=float(x)
  if not math.isfinite(v):raise ValueError('nonfinite JSON float '+x)
  return v
 return json.loads(text,object_pairs_hook=pairs,parse_constant=bad,parse_float=real)

def num(x):
 try:return type(x) in (int,float) and math.isfinite(x) and x>=0
 except OverflowError:return False

def read_json(p,issues,label,partial=False):
 if p is None or not p.exists():return None,'absent'
 b=p.read_bytes()
 if not b.strip():return None,'empty'
 try:return strict_json(b.decode('utf8')),'valid'
 except (ValueError,UnicodeDecodeError) as e:
  if partial:return None,'partial-or-invalid-while-live'
  issues.append(label+' invalid JSON: '+str(e));return None,'invalid'

def lines(p):
 if p is None or not p.exists():return [],False
 seq=p.read_text(errors='replace').splitlines(keepends=True);partial=bool(seq and not seq[-1].endswith('\n'))
 if partial:seq.pop()
 return [x.rstrip('\r\n') for x in seq],partial

def filehash(p):return hashlib.sha256(p.read_bytes()).hexdigest() if p is not None and p.exists() else None

def analyze_telemetry(audit,term,pidmap,uuid,max_temp,issues):
 values=[];pids=set();regpids=set();puuids=set();ruuids=set();foreign=0
 for i,a in enumerate(audit,1):
  try:
   t=float(a['elapsed_s']);s=a['sample']
   if len(s)!=4:raise ValueError('node2 sample must be temperature,memory,utilization,power')
   temp=int(s[0]);mem=int(s[1]);util=float(s[2]);power=float(s[3])
   if not all(num(x) for x in (t,temp,mem,util,power)):raise ValueError('negative/nonfinite')
   values.append((t,temp,mem,util,power));foreign+=bool(a['foreign'])
   for q in a['processes']:pids.add(int(q['pid']));puuids.add(q['uuid'])
   if a.get('registered'):
    q=a['registered'];regpids.add(int(q['nvml_host_pid']));ruuids.add(q['gpu_uuid'])
    if isinstance(pidmap,dict) and q!=pidmap:issues.append('sample registered map mismatch')
  except (KeyError,ValueError,TypeError,IndexError) as e:issues.append('gpu sample '+str(i)+': '+str(e))
 if not values:return {'samples':0,'noncontinuous':True,'foreign_samples':foreign}
 ts=[x[0] for x in values];gaps=[b-a for a,b in zip(ts,ts[1:])]
 if any(g<0 for g in gaps):issues.append('nonmonotonic telemetry')
 if foreign:issues.append('foreign GPU process sample')
 if puuids-{uuid} or ruuids-{uuid}:issues.append('telemetry GPU UUID mismatch')
 if max_temp is not None and max(x[1] for x in values)>max_temp:issues.append('sampled temperature exceeds supplied guard limit')
 pid=pidmap.get('nvml_host_pid') if isinstance(pidmap,dict) else None
 if pid is not None and ((pids and pids!={pid}) or (regpids and regpids!={pid})):issues.append('registered/observed PID mismatch')
 wall=term.get('wall_seconds') if isinstance(term,dict) else None
 tail=wall-ts[-1] if num(wall) and wall>=ts[-1] else None
 return {'samples':len(values),'noncontinuous':True,'start_to_first_sample_s':ts[0],'max_gap_between_samples_s':max(gaps) if gaps else None,'last_sample_to_terminal_s':tail,'temperature_min_C':min(x[1] for x in values),'temperature_max_C':max(x[1] for x in values),'max_sampled_memory_MiB':max(x[2] for x in values),'max_sampled_utilization_percent':max(x[3] for x in values),'max_sampled_power_W':max(x[4] for x in values),'sm_clock_MHz_when_nonzero_util':None,'memory_clock_MHz_when_nonzero_util':None,'clocks_scope':'not sampled by the node2 guard; no inferred clocks','foreign_samples':foreign,'observed_host_pids':sorted(pids),'registered_host_pids':sorted(regpids),'observed_uuids':sorted(puuids),'registered_uuids':sorted(ruuids)}

def check_refresh(x,issues,where):
 if not isinstance(x,dict):issues.append(where+' not object');return False
 for k in ('fullslot_identity_max_complex','fullslot_real_sine_max_complex'):
  if not num(x.get(k)):issues.append(where+' invalid/nonfinite '+k)
 for k,v in (('Q_in',6),('input_log_scale',40),('Q_out',27),('output_log_scale',50)):
  if x.get(k)!=v:issues.append(where+' metadata mismatch '+k)
 if x.get('immutable') is not True:issues.append(where+' immutable false/absent')
 return all(num(x.get(k)) and x[k]<=THRESHOLD for k in ('fullslot_identity_max_complex','fullslot_real_sine_max_complex'))

def check_case(x,i,refreshes,issues):
 where='case '+str(x.get('name','?'))
 if x.get('name')!=('a_b','swapped_b_a')[i]:issues.append(where+' name/order mismatch')
 for k,v in (('physical_Boots',16),('projection_pairs',8),('derived_EvalMod_calls',32),('final_Q',10),('final_log_scale',40)):
  if x.get(k)!=v:issues.append(where+' mismatch '+k)
 for k in ('paired_observer_OFF_ON_metadata_RNS_exact','all_inputs_const_decodes_and_outputs_immutable','all_slots_finite','diagnostic_includes_control_snapshots_and_observer_decode','maxima_not_additive'):
  if x.get(k) is not True:issues.append(where+' false/absent '+k)
 if not num(x.get('diagnostic_seconds')):issues.append(where+' diagnostic seconds invalid')
 gates=[]
 for name in OUTPUTS:
  e=x.get(name)
  if not isinstance(e,dict):issues.append(where+' missing '+name);continue
  for k in ('active38_max_complex_source','fullslot_max_complex_source','max_imaginary'):
   if not num(e.get(k)):issues.append(where+' '+name+' invalid '+k)
  for k in COND:
   if k not in e:issues.append(where+' '+name+' missing conditioned field '+k)
   elif name.startswith('paired_') and not num(e[k]):issues.append(where+' '+name+' invalid conditioned metric '+k)
   elif name.startswith('old_') and e[k] is not None:issues.append(where+' '+name+' conditioned must be null '+k)
  a=e.get('active38_max_complex_source');f=e.get('fullslot_max_complex_source');imag=e.get('max_imaginary')
  if num(a) and num(f):
   if a>f+1e-14:issues.append(where+' '+name+' active exceeds full')
   if num(imag) and imag>f+1e-14:issues.append(where+' '+name+' imaginary exceeds full')
   ap=a<=THRESHOLD;fp=f<=THRESHOLD
   if e.get('active38_source_pass') is not ap or e.get('fullslot_source_pass') is not fp:issues.append(where+' '+name+' source gate mismatch')
   gates.extend((ap,fp))
 if x.get('refreshes')!=[z['data'] for z in refreshes]:issues.append(where+' stdout vs case refresh rows mismatch')
 refresh_pass=[check_refresh(z['data'],issues,where+' refresh '+str(k)) for k,z in enumerate(refreshes)]
 if len(gates)==8 and len(refreshes)==8:
  passed=all(gates) and all(refresh_pass)
  if x.get('all_fixed_gates_pass') is not passed:issues.append(where+' final gate mismatch')
 return len(gates)==8 and len(refreshes)==8 and all(gates) and all(refresh_pass)

def analyze_postflight(post,uuid,issues):
 if not isinstance(post,dict):issues.append('postflight not object');return None
 expected={'pid':GUARD_PID,'pgid':GUARD_PID,'start_ticks':GUARD_START}
 if post.get('expected_gpu_uuid')!=uuid or post.get('expected_guard_identity')!=expected:issues.append('postflight expected identity mismatch')
 guard_absent='guard_identity_once' in post and post['guard_identity_once'] is None
 child_absent='registered_namespace_identity_once' in post and post['registered_namespace_identity_once'] is None
 gpu=post.get('gpu');apps=post.get('compute_apps');idle=False
 try:
  rows=gpu['rows']
  idle=(gpu['returncode']==0 and len(rows)==1 and len(rows[0])==5 and rows[0][0]==uuid and int(rows[0][3])==0 and num(int(rows[0][4])) and apps['returncode']==0 and apps['rows']==[])
 except (KeyError,TypeError,ValueError,IndexError):pass
 checks={'expected_guard_absent':guard_absent,'registered_namespace_child_absent':child_absent,'target_gpu_zero_MiB_no_compute_PID':idle}
 for k,v in checks.items():
  if not v:issues.append('postflight not established '+k)
 return checks

def analyze(paths,uuid,start_temp=None,max_temp=None):
 issues=[];failures=[];setups=[];cases=[];refresh=[];events=[];dones=[];cpu=[];common=[];registrations=[]
 if uuid!=EXPECTED_UUID:issues.append('expected node2 GPU UUID mismatch')
 if not num(start_temp) or not num(max_temp) or start_temp>max_temp:issues.append('valid explicit guard temperature bounds required')
 stdout,stdout_partial=lines(paths.get('stdout'))
 for n,line in enumerate(stdout,1):
  names=('PUBLIC_SILU_PAIR_SETUP ','PUBLIC_SILU_PAIR_REFRESH ','PUBLIC_SILU_PAIR_CASE ')
  if line.startswith(names):
   tag,text=line.split(' ',1)
   try:z=strict_json(text)
   except ValueError as e:failures.append({'line':n,'error':tag+' JSON '+str(e)});continue
   if not isinstance(z,dict):failures.append({'line':n,'error':tag+' non-object'});continue
   (setups if tag.endswith('SETUP') else refresh if tag.endswith('REFRESH') else cases).append({'stdout_line':n,'data':z})
  elif line.startswith('A800_PID_REGISTRATION '):
   try:
    z=strict_json(line.split(' ',1)[1])
    if not isinstance(z,dict):raise ValueError('not object')
    registrations.append({'stdout_line':n,'data':z})
   except ValueError as e:failures.append({'line':n,'error':'invalid PID registration '+str(e)})
  elif line.startswith('PUBLIC_SILU_PAIR_CPU_DONE '):cpu.append({'stdout_line':n,'raw':line})
  elif line.startswith('PUBLIC_PROJECTIONS_CPU_DONE '):common.append({'stdout_line':n,'raw':line})
  elif line.startswith('PUBLIC_SILU_PAIR_DONE '):dones.append({'stdout_line':n,'raw':line})
  elif line.startswith('PUBLIC_SILU_PAIR_FAILED '):failures.append({'line':n,'error':line})
  elif line.startswith('SOURCE_BOOT_PROJECTIONS_CANDIDATE '):
   m=re.fullmatch(r'SOURCE_BOOT_PROJECTIONS_CANDIDATE physical_boot_calls=(\d+) pairs=(\d+) component_outputs=2 semantic_qualification=false output_q=27 log_scale=50',line)
   if m:events.append({'stdout_line':n,'kind':'pair','calls':int(m.group(1)),'pairs':int(m.group(2))})
   else:failures.append({'line':n,'error':'malformed projection counter'})
  elif line.startswith('SOURCE_BOOT point='):
   m=re.fullmatch(r'SOURCE_BOOT point=(silu_internal:b33_2|silu60:g2_refresh[012]) input_q=6 output_q=27 log_scale=50 calls=(\d+) corrections=0',line)
   if m:events.append({'stdout_line':n,'kind':m.group(1),'calls':int(m.group(2))})
   else:failures.append({'line':n,'error':'unexpected/malformed source Boot counter'})
 term,ts=read_json(paths.get('terminal'),issues,'terminal',True);live=ts!='valid'
 command,cs=read_json(paths.get('command'),issues,'command',live);pidmap,ps=read_json(paths.get('pid_map'),issues,'pid map',live);summary,ss=read_json(paths.get('summary'),issues,'SUMMARY',live);postflight,posts=read_json(paths.get('postflight'),issues,'postflight',live)
 guard_launch,gls=read_json(paths.get('guard_launch'),issues,'guard launch log',live)
 if guard_launch is not None and isinstance(term,dict) and guard_launch!=term:issues.append('guard launch log vs terminal mismatch')
 postchecks=analyze_postflight(postflight,uuid,issues) if postflight is not None else None
 if not live and not isinstance(postflight,dict):issues.append('terminal requires postflight evidence')
 if not live and not isinstance(guard_launch,dict):issues.append('terminal requires guard final printed record')
 gpu,gpu_partial=lines(paths.get('gpu'));audit=[]
 for n,line in enumerate(gpu,1):
  try:audit.append(strict_json(line))
  except ValueError as e:issues.append('gpu.jsonl '+str(n)+' invalid: '+str(e))
 if not live and (stdout_partial or gpu_partial):issues.append('terminal capture has incomplete final stdout/gpu line')
 tel=analyze_telemetry(audit,term,pidmap,uuid,max_temp,issues)
 if len(cpu)>1 or len(common)>1 or len(registrations)>1 or len(setups)>1 or len(dones)>1 or len(cases)>2 or len(refresh)>16 or len(events)>32:issues.append('duplicate/excess records')
 if registrations and isinstance(pidmap,dict) and registrations[0]['data']!=pidmap:issues.append('stdout vs PID map registration mismatch')
 if cpu and cpu[0]['raw']!=CPU_MARKER:issues.append('SiLU CPU marker mismatch')
 if common and common[0]['raw']!=COMMON_MARKER:issues.append('common CPU schema marker mismatch')
 if setups:
  z=setups[0]['data']
  for k,v in (('N',65536),('complex_slots',32768),('ratio',65536),('m',2),('native_q',27),('output_log_scale',50),('performance_valid',False)):
   if z.get(k)!=v or (type(v) is bool and type(z.get(k)) is not bool):issues.append('setup mismatch '+k)
  for k in ('eta','qdiff','runtime_profile_setup_seconds','source_vs_true_SiLU_max_complex'):
   if not num(z.get(k)):issues.append('setup invalid '+k)
  if num(z.get('eta')) and z['eta']==0 or num(z.get('qdiff')) and z['qdiff']==0:issues.append('setup eta/qdiff nonpositive')
 pattern=('pair',)*8+('silu_internal:b33_2','silu60:g2_refresh0','silu60:g2_refresh1','silu60:g2_refresh2')*2
 pairs=0
 for i,e in enumerate(events):
  if e['calls']!=i+1 or e['kind']!=pattern[i%16]:issues.append('actual Boot event order/counter mismatch')
  if e['kind']=='pair':
   pairs+=1
   if e['pairs']!=pairs:issues.append('actual projection pair counter mismatch')
 # Check all rows even when a structural failure prevents a CASE record.
 all_refresh_pass=[]
 for j,r in enumerate(refresh):
  z=r['data'];all_refresh_pass.append(check_refresh(z,issues,'observed refresh '+str(j)))
  if type(z.get('ordinal')) is not int or type(z.get('lane')) is not int or (z.get('ordinal'),z.get('lane'))!=((j%8)//2,j%2):issues.append('observed refresh ordinal/lane order mismatch')
  preceding=[e for e in events if e['stdout_line']<r['stdout_line']]
  expected_call=(j//8)*16+5+(j%8)//2
  if not preceding or preceding[-1]['kind']!='pair' or preceding[-1]['calls']!=expected_call:issues.append('observed refresh not attached to corresponding ON candidate counter')
 case_pass=[];prior_line=setups[0]['stdout_line'] if setups else 0
 for i,c in enumerate(cases):
  if i>=2:break
  case_events=[e for e in events if prior_line<e['stdout_line']<c['stdout_line']]
  case_refresh=[r for r in refresh if prior_line<r['stdout_line']<c['stdout_line']]
  if len(case_events)!=16 or [e['calls'] for e in case_events]!=list(range(i*16+1,i*16+17)):issues.append('case '+str(i)+' lacks 16 observed Boot events')
  if len(case_refresh)!=8:issues.append('case '+str(i)+' lacks 8 observed refresh rows')
  for j,r in enumerate(case_refresh):
   z=r['data']
   if type(z.get('ordinal')) is not int or type(z.get('lane')) is not int or (z.get('ordinal'),z.get('lane'))!=(j//2,j%2):issues.append('case refresh ordinal/lane order mismatch')
   if len(case_events)==16 and not (case_events[4+j//2]['stdout_line']<r['stdout_line']<case_events[5+j//2]['stdout_line']):issues.append('refresh not attached to corresponding ON candidate counter')
  case_pass.append(check_case(c['data'],i,case_refresh,issues));prior_line=c['stdout_line']
 overall=all(case_pass) if len(case_pass)==2 else None
 done_pass=None;done_boots=None;done_pairs=None
 if dones:
  m=re.fullmatch(r'PUBLIC_SILU_PAIR_DONE numerical_pass=([01]) physical_Boots=(\d+) projection_pairs=(\d+)',dones[0]['raw'])
  if not m:issues.append('DONE syntax mismatch')
  else:done_pass=m.group(1)=='1';done_boots=int(m.group(2));done_pairs=int(m.group(3))
  if (done_boots,done_pairs)!=(32,16):issues.append('DONE actual counts mismatch')
  if overall is not None and done_pass is not overall:issues.append('DONE vs recomputed gate mismatch')
 if isinstance(summary,dict):
  expected={'status':'completed_public_silu_pair_control','N':65536,'complex_slots':32768,'active_tokens':38,'case_count':2,'physical_Boots':32,'projection_pairs':16,'derived_internal_EvalMod_calls':64,'model_qualification':False,'performance_valid':False}
  for k,v in expected.items():
   if summary.get(k)!=v or (type(v) is bool and type(summary.get(k)) is not bool):issues.append('SUMMARY mismatch '+k)
  for k in ('runtime_profile_setup_seconds','source_vs_true_SiLU_max_complex'):
   if not num(summary.get(k)) or (setups and summary.get(k)!=setups[0]['data'].get(k)):issues.append('SUMMARY invalid/different '+k)
  if summary.get('cases')!=[c['data'] for c in cases]:issues.append('SUMMARY vs stdout cases mismatch')
  if overall is not None and summary.get('all_numerical_gates_passed') is not overall:issues.append('SUMMARY gate mismatch')
 elif summary is not None:issues.append('SUMMARY not object')
 for label,g in (('command',command),('terminal',term)):
  if isinstance(g,dict):
   gpu=g.get('initial_gpu');actual=gpu[0].strip() if isinstance(gpu,list) and len(gpu)==5 and isinstance(gpu[0],str) else None
   try:initial=int(gpu[4]);initial_memory=int(gpu[3])
   except (TypeError,ValueError,IndexError):initial=None;initial_memory=None
   if actual!=uuid or not isinstance(g.get('env'),dict) or g['env'].get('CUDA_VISIBLE_DEVICES')!=uuid or not num(initial) or initial_memory!=0:issues.append(label+' UUID/initial temperature/memory mismatch')
   elif start_temp is not None and initial>start_temp:issues.append(label+' initial temperature exceeds supplied guard limit')
   if g.get('authorization_sha256')!=AUTH_SHA or g.get('guard_pid')!=GUARD_PID or g.get('command')!=EXPECTED_COMMAND:issues.append(label+' authorization/guard/closed command mismatch')
   if not isinstance(g.get('env'),dict) or any(g['env'].get(k)!=v for k,v in EXPECTED_ENV.items()):issues.append(label+' authorized environment mismatch')
   leases=g.get('owned_shared_leases')
   if not isinstance(leases,dict) or len(leases)!=8 or any(not isinstance(v,dict) or v.get('uid')!=0 or not num(v.get('inode')) or not num(v.get('mode')) for v in leases.values()):issues.append(label+' eight owned existing lease records invalid')
 if isinstance(term,dict):
  for k in ('returncode','abort_reason','requested_signals','wall_seconds','registered','classification','child_identity','end'):
   if k not in term:issues.append('terminal missing '+k)
  if 'returncode' in term and type(term['returncode']) is not int:issues.append('guard invalid return code type')
  if term.get('abort_reason') is not None and not isinstance(term.get('abort_reason'),str):issues.append('guard invalid abort reason type')
  if term.get('requested_signals')!=[]:issues.append('guard signal or invalid signal list')
  if term.get('cleanup_error') or term.get('owned_process_may_still_be_live'):issues.append('guard cleanup unresolved')
  if not num(term.get('wall_seconds')):issues.append('guard wall invalid')
  rc0=term.get('returncode');abort0=term.get('abort_reason')
  expected_class='numerical_negative' if rc0==2 and abort0 is None else ('completed' if rc0==0 and abort0 is None else 'structural_or_resource_failure')
  if term.get('classification')!=expected_class:issues.append('guard classification mismatch')
  if term.get('registered')!=pidmap:issues.append('terminal vs PID map registration mismatch')
  child=term.get('child_identity')
  if isinstance(pidmap,dict):
   if not isinstance(child,dict) or any(child.get(k)!=pidmap.get(v) for k,v in (('pid','namespace_pid'),('pgid','pgid'),('start_ticks','start_ticks'))) or child.get('sid')!=pidmap.get('pgid') or child.get('uid')!=0:issues.append('terminal child identity mismatch')
 if isinstance(command,dict) and isinstance(term,dict):
  for k in ('initial_gpu','env','start','command','guard_pid','authorization_sha256','owned_shared_leases'):
   if k in command and command.get(k)!=term.get(k):issues.append('initial/terminal mismatch '+k)
 if isinstance(pidmap,dict) and (pidmap.get('gpu_uuid')!=uuid or pidmap.get('supervisor_pid')!=GUARD_PID or pidmap.get('namespace_pid')!=pidmap.get('pgid') or not all(type(pidmap.get(k)) is int and pidmap[k]>0 for k in ('nvml_host_pid','namespace_pid','pgid','supervisor_pid')) or not str(pidmap.get('start_ticks','')).isdigit()):issues.append('registration UUID/PID/start mismatch')
 rc=term.get('returncode') if isinstance(term,dict) else None;abort=term.get('abort_reason') if isinstance(term,dict) else None
 pid=pidmap.get('nvml_host_pid') if isinstance(pidmap,dict) else None
 pid_valid=pid is not None and tel.get('registered_host_pids')==tel.get('observed_host_pids')==[pid]
 complete=bool(len(cpu)==len(common)==len(registrations)==len(setups)==len(dones)==1 and len(cases)==2 and len(refresh)==16 and len(events)==32 and pairs==16 and (done_boots,done_pairs)==(32,16) and overall is not None and rc==(0 if overall else 2) and abort is None and isinstance(term,dict) and isinstance(command,dict) and isinstance(summary,dict) and pid_valid and not issues and not failures)
 if complete:status='complete-pass' if overall else 'complete-numerical-negative'
 elif abort is not None:status='guard-abort'
 elif (rc is not None and rc not in (0,2)) or failures:status='structural-or-app-failure'
 elif issues:status='incomplete-or-inconsistent-evidence'
 else:status='incomplete'
 if failures:stage=failures[-1]['error']
 elif not setups:stage='before paired SiLU profile setup'
 elif len(cases)<2:stage='case '+str(len(cases)+1)+' of 2 incomplete'
 elif not dones:stage='after cases before DONE'
 elif ss!='valid':stage='SUMMARY absent/empty/invalid'
 elif not complete:stage='terminal, telemetry or validation incomplete'
 else:stage=None
 return {'schema_version':2,'status':status,'failure_stage':stage,'complete_diagnostic':complete,'numerical_gates_passed':overall,'complete_finite_numeric_negative_rc2':bool(complete and not overall and rc==2),'capture_states':{'stdout':'present' if paths.get('stdout') and paths['stdout'].exists() else 'absent','stdout_partial_line':stdout_partial,'gpu_partial_line':gpu_partial,'command':cs,'terminal':ts,'pid_map':ps,'summary':ss,'postflight':posts,'guard_launch':gls},'setup':setups,'cases':cases,'refresh_records':refresh,'boot_counter_events':events,'done':dones,'stdout_registrations':registrations,'silu_cpu_marker':cpu,'common_cpu_marker':common,'summary':summary,'command':command,'terminal':term,'pid_map':pidmap,'telemetry':tel,'postflight':postflight,'postflight_checks':postchecks,'guard_launch_terminal_record':guard_launch,'child_returncode':rc,'outer_classification':term.get('classification') if isinstance(term,dict) else None,'outer_exit_status_scope':'not independently waitpid-observed by detached monitor; guard classification and final printed record only','observed_refresh_checks':{'rows':len(refresh),'all_observed_fixed_refresh_gates_pass':all(all_refresh_pass) if refresh else None,'identity_max_complex':max((r['data']['fullslot_identity_max_complex'] for r in refresh if num(r['data'].get('fullslot_identity_max_complex'))),default=None),'real_sine_max_complex':max((r['data']['fullslot_real_sine_max_complex'] for r in refresh if num(r['data'].get('fullslot_real_sine_max_complex'))),default=None),'scope':'observed prefix only; not final source-oracle or whole-control qualification'},'timings':{'runtime_profile_setup_seconds':setups[0]['data'].get('runtime_profile_setup_seconds') if setups else None,'completed_case_diagnostic_seconds':[c['data'].get('diagnostic_seconds') for c in cases],'incomplete_case_diagnostic_seconds':None,'guard_wall_seconds':term.get('wall_seconds') if isinstance(term,dict) else None,'performance_valid':False},'final_validation':{'closed_shape_predicate_failed':any('projection actual N/Q/id/scale/components/device mismatch' in f['error'] for f in failures),'shape_failure_detail_scope':'generic predicate only; actual failing output and individual metadata field were not logged','OFF_ON_same_plan_metadata_RNS_exact':all(c['data'].get('paired_observer_OFF_ON_metadata_RNS_exact') is True for c in cases) if len(cases)==2 else None,'all_inputs_const_decodes_and_outputs_immutable':all(c['data'].get('all_inputs_const_decodes_and_outputs_immutable') is True for c in cases) if len(cases)==2 else None,'all_slots_finite':all(c['data'].get('all_slots_finite') is True for c in cases) if len(cases)==2 else None,'final_Q10_scale2pow40_metadata':True if len(cases)==2 and all(c['data'].get('final_Q')==10 and c['data'].get('final_log_scale')==40 for c in cases) else None},'issues':issues,'failures':failures,'observed_counts':{'case_rows':len(cases),'refresh_rows':len(refresh),'physical_Boot_counter_events':len(events),'projection_pair_counter_events':pairs,'last_physical_Boot_counter':events[-1]['calls'] if events else None,'last_projection_pair_counter':next((e['pairs'] for e in reversed(events) if e['kind']=='pair'),None),'physical_Boots_from_DONE':done_boots,'projection_pairs_from_DONE':done_pairs,'internal_EvalMod_calls_derived_not_independently_counted':summary.get('derived_internal_EvalMod_calls') if complete else None},'scope':'public paired-SiLU one process; no Native-identical FHE/model/Torch/security/performance qualification','refresh_oracles':'identity uses full complex preinput; real_sine uses factor*sin(real(preinput)/factor) with complex output error, NOT full complex-sine oracle','conditioned_metrics':'paired conditioned metrics are non-gating, old control conditioned metrics must be null; independent maxima not additive','timing_scope':'diagnostic wall includes control snapshots and observer decode; GPU busy/clean performance not measured','source_vs_true_SiLU_max_complex':setups[0]['data'].get('source_vs_true_SiLU_max_complex') if setups else None,'source_vs_true_SiLU_scope':'separate non-gating nominal source vs mathematical SiLU comparison, not Torch qualification','telemetry_scope':'noncontinuous sampled telemetry; gaps and unsampled tail explicitly reported','postflight_scope':'one explicit sampled GPU/process observation; no continuous exclusivity or lease-release proof','hashes':{k:filehash(v) for k,v in paths.items()}}

def write_outputs(output,result):
 output=pathlib.Path(output);output.mkdir(parents=True,exist_ok=True)
 (output/'PARSED.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 fields=['name','physical_Boots','projection_pairs','derived_EvalMod_calls','final_Q','final_log_scale','paired_observer_OFF_ON_metadata_RNS_exact','all_inputs_const_decodes_and_outputs_immutable','all_slots_finite','diagnostic_seconds','all_fixed_gates_pass']+[f'{o}_{k}' for o in OUTPUTS for k in ('active38_max_complex_source','fullslot_max_complex_source','max_imaginary',*COND,'active38_source_pass','fullslot_source_pass')]
 with (output/'CASES.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for item in result['cases']:
   z=item['data'];row={k:z.get(k) for k in fields}
   for o in OUTPUTS:
    v=z.get(o) if isinstance(z.get(o),dict) else {}
    for k in ('active38_max_complex_source','fullslot_max_complex_source','max_imaginary',*COND,'active38_source_pass','fullslot_source_pass'):row[f'{o}_{k}']=v.get(k)
   w.writerow(row)
 fields=['case_index','stdout_line','ordinal','lane','fullslot_identity_max_complex','fullslot_real_sine_max_complex','Q_in','input_log_scale','Q_out','output_log_scale','immutable']
 with (output/'REFRESHES.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for z in result['refresh_records']:
   row={k:z['data'].get(k) for k in fields};row['stdout_line']=z['stdout_line'];row['case_index']=sum(c['stdout_line']<z['stdout_line'] for c in result['cases']);w.writerow(row)
 (output/'FAIL.json').write_text(json.dumps({k:result[k] for k in ('status','complete_diagnostic','failure_stage','issues','failures') if k in result},indent=2)+'\n')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('stdout','command','terminal','gpu','pid-map','summary','postflight','guard-launch'):p.add_argument('--'+name,type=pathlib.Path)
 p.add_argument('--gpu-uuid',required=True);p.add_argument('--start-temp',type=float,required=True);p.add_argument('--max-temp',type=float,required=True);p.add_argument('--no-write',action='store_true');a=p.parse_args()
 paths={k:getattr(a,k) for k in ('stdout','command','terminal','gpu','pid_map','summary','postflight','guard_launch')};v=analyze(paths,a.gpu_uuid,a.start_temp,a.max_temp)
 if not a.no_write:write_outputs(HERE,v)
 print(json.dumps({'status':v['status'],'complete':v['complete_diagnostic'],'numerical_gates_passed':v['numerical_gates_passed'],'observed_counts':v['observed_counts'],'issues':len(v['issues']),'failures':len(v['failures'])}))
if __name__=='__main__':main()
