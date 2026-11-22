#!/usr/bin/env python3
"""Pure synthetic aggregate fixtures: no app, remote system, GPU, CT/PT or keys."""
import copy,json,pathlib,tempfile,unittest
import parse
UUID=parse.EXPECTED_UUID

def source(old=False,active=0.0001,full=0.0002):
 return {'active38_max_complex_source':active,'fullslot_max_complex_source':full,'max_imaginary':0.00001,'post_Gate_refresh_decodedUp_conditioned':None if old else 0.25,'post_all_refresh_decodedUp_conditioned':None if old else 0.4,'active38_source_pass':active<=.001,'fullslot_source_pass':full<=.001}

def fixture(d,negative=None):
 result=d/'guard';report=d/'app-report';result.mkdir();report.mkdir();mapping={'nvml_host_pid':100,'namespace_pid':200,'pgid':200,'start_ticks':'1234','gpu_uuid':UUID,'supervisor_pid':parse.GUARD_PID};rows=[];log=['A800_PID_REGISTRATION '+json.dumps(mapping),parse.COMMON_MARKER,parse.CPU_MARKER]
 setup={'N':65536,'complex_slots':32768,'ratio':65536,'m':2,'eta':1.03,'qdiff':.98,'native_q':27,'output_log_scale':50,'runtime_profile_setup_seconds':100.0,'source_vs_true_SiLU_max_complex':.495,'performance_valid':False}
 log.append('PUBLIC_SILU_PAIR_SETUP '+json.dumps(setup))
 for c in range(2):
  refs=[]
  for j in range(8):
   call=c*16+j+1;pair=c*8+j+1
   log.append(f'SOURCE_BOOT_PROJECTIONS_CANDIDATE physical_boot_calls={call} pairs={pair} component_outputs=2 semantic_qualification=false output_q=27 log_scale=50')
   if j>=4:
    for lane in range(2):
     x={'ordinal':j-4,'lane':lane,'fullslot_identity_max_complex':.0001,'fullslot_real_sine_max_complex':.0002,'Q_in':6,'input_log_scale':40,'Q_out':27,'output_log_scale':50,'immutable':True}
     if negative=='refresh' and c==1 and j==5 and lane==1:x['fullslot_real_sine_max_complex']=.0012
     refs.append(x);log.append('PUBLIC_SILU_PAIR_REFRESH '+json.dumps(x))
  oldkinds=('silu_internal:b33_2','silu60:g2_refresh0','silu60:g2_refresh1','silu60:g2_refresh2')*2
  for j,k in enumerate(oldkinds):log.append(f'SOURCE_BOOT point={k} input_q=6 output_q=27 log_scale=50 calls={c*16+9+j} corrections=0')
  row={'name':('a_b','swapped_b_a')[c],'physical_Boots':16,'projection_pairs':8,'derived_EvalMod_calls':32,'paired_lane0':source(),'paired_lane1':source(),'old_lane0':source(True),'old_lane1':source(True),'paired_observer_OFF_ON_metadata_RNS_exact':True,'all_inputs_const_decodes_and_outputs_immutable':True,'all_slots_finite':True,'final_Q':10,'final_log_scale':40,'diagnostic_includes_control_snapshots_and_observer_decode':True,'diagnostic_seconds':42.0,'maxima_not_additive':True,'all_fixed_gates_pass':not (negative and c==1),'refreshes':refs}
  if negative=='full' and c==1:row['old_lane1']=source(True,full=.0012)
  rows.append(row);log.append('PUBLIC_SILU_PAIR_CASE '+json.dumps(row))
 passed=not negative
 summary={'status':'completed_public_silu_pair_control','N':65536,'complex_slots':32768,'active_tokens':38,'case_count':2,'physical_Boots':32,'projection_pairs':16,'derived_internal_EvalMod_calls':64,'runtime_profile_setup_seconds':100.0,'source_vs_true_SiLU_max_complex':.495,'all_numerical_gates_passed':passed,'model_qualification':False,'performance_valid':False,'cases':rows}
 log.append('PUBLIC_SILU_PAIR_DONE numerical_pass='+('1' if passed else '0')+' physical_Boots=32 projection_pairs=16')
 (result/'stdout.log').write_text('\n'.join(log)+'\n');(report/'SUMMARY.json').write_text(json.dumps(summary))
 terminal={'initial_gpu':[UUID,'NVIDIA A800 80GB PCIe','81920','0','34'],'env':{'CUDA_VISIBLE_DEVICES':UUID,**parse.EXPECTED_ENV},'returncode':0 if passed else 2,'abort_reason':None,'requested_signals':[],'wall_seconds':6.0,'start':'start','end':'end','authorization_sha256':parse.AUTH_SHA,'guard_pid':parse.GUARD_PID,'command':parse.EXPECTED_COMMAND,'owned_shared_leases':{str(i):{'inode':i+1,'mode':33188,'uid':0} for i in range(8)},'child_identity':{'pid':200,'pgid':200,'sid':200,'start_ticks':'1234','uid':0},'registered':mapping,'classification':'completed' if passed else 'numerical_negative'}
 (result/'terminal.json').write_text(json.dumps(terminal));(result/'guard-launch.log').write_text(json.dumps(terminal));(result/'command.json').write_text(json.dumps({k:terminal[k] for k in ('initial_gpu','env','start','authorization_sha256','guard_pid','command','owned_shared_leases')}))
 post={'expected_gpu_uuid':UUID,'expected_guard_identity':{'pid':parse.GUARD_PID,'pgid':parse.GUARD_PID,'start_ticks':parse.GUARD_START},'guard_identity_once':None,'registered_namespace_identity_once':None,'gpu':{'returncode':0,'rows':[[UUID,'NVIDIA A800 80GB PCIe','81920','0','29']]},'compute_apps':{'returncode':0,'rows':[]}}
 (result/'postflight.json').write_text(json.dumps(post))
 (result/'pid-map.json').write_text(json.dumps(mapping))
 samples=[{'elapsed_s':1.0,'sample':['34','0','0','48'],'processes':[],'foreign':[],'registered':None},{'elapsed_s':3.0,'sample':['40','1000','50','180'],'processes':[{'pid':'100','uuid':UUID}],'foreign':[],'registered':mapping}]
 (result/'gpu.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in samples))
 return {'stdout':result/'stdout.log','summary':report/'SUMMARY.json','command':result/'command.json','terminal':result/'terminal.json','pid_map':result/'pid-map.json','gpu':result/'gpu.jsonl','postflight':result/'postflight.json','guard_launch':result/'guard-launch.log'}

class ParserTest(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory(dir=parse.HERE,prefix='parser-test-');self.d=pathlib.Path(self.temp.name)
 def tearDown(self):self.temp.cleanup()
 def parse(self,p):return parse.analyze(p,UUID,55,75)
 def test_pass_and_real_counters_32_16(self):
  p=fixture(self.d);v=self.parse(p);self.assertEqual(v['status'],'complete-pass',v['issues']);self.assertEqual(v['observed_counts']['physical_Boot_counter_events'],32);self.assertEqual(v['observed_counts']['projection_pair_counter_events'],16);self.assertEqual(v['observed_counts']['internal_EvalMod_calls_derived_not_independently_counted'],64);self.assertEqual(len(v['refresh_records']),16);self.assertEqual(v['telemetry']['max_gap_between_samples_s'],2);self.assertIsNone(v['telemetry']['sm_clock_MHz_when_nonzero_util']);self.assertEqual(v['setup'][0]['data']['source_vs_true_SiLU_max_complex'],.495)
  parse.write_outputs(self.d,v);self.assertEqual(len((self.d/'CASES.csv').read_text().splitlines()),3);self.assertEqual(len((self.d/'REFRESHES.csv').read_text().splitlines()),17)
 def test_complete_finite_rc2_only_fullsource(self):
  v=self.parse(fixture(self.d,'full'));self.assertEqual(v['status'],'complete-numerical-negative',v['issues']);self.assertTrue(v['complete_finite_numeric_negative_rc2']);self.assertTrue(v['cases'][1]['data']['old_lane1']['active38_source_pass']);self.assertFalse(v['cases'][1]['data']['old_lane1']['fullslot_source_pass'])
 def test_complete_finite_rc2_only_refresh(self):
  v=self.parse(fixture(self.d,'refresh'));self.assertEqual(v['status'],'complete-numerical-negative',v['issues']);self.assertFalse(v['cases'][1]['data']['all_fixed_gates_pass']);self.assertTrue(all(v['cases'][1]['data'][o]['fullslot_source_pass'] for o in parse.OUTPUTS))
 def test_rns_fail_closed(self):
  p=fixture(self.d);p['stdout'].write_text(p['stdout'].read_text().replace('"paired_observer_OFF_ON_metadata_RNS_exact": true','"paired_observer_OFF_ON_metadata_RNS_exact": false',1));v=self.parse(p);self.assertFalse(v['complete_diagnostic']);self.assertTrue(any('RNS' in e for e in v['issues']))
 def test_pid_foreign_fail_closed(self):
  p=fixture(self.d);a=[json.loads(x) for x in p['gpu'].read_text().splitlines()];a[1]['processes'][0]['pid']='101';a[1]['foreign']=[{'pid':'101','uuid':UUID}];p['gpu'].write_text(''.join(json.dumps(x)+'\n' for x in a));v=self.parse(p);self.assertFalse(v['complete_diagnostic']);self.assertTrue(any('foreign' in x for x in v['issues']));self.assertTrue(any('PID mismatch' in x for x in v['issues']))
 def test_registration_stdout_mismatch(self):
  p=fixture(self.d);mapping=json.loads(p['pid_map'].read_text());mapping['supervisor_pid']=98;p['pid_map'].write_text(json.dumps(mapping));v=self.parse(p);self.assertFalse(v['complete_diagnostic']);self.assertIn('stdout vs PID map registration mismatch',v['issues'])
 def test_missing_abort_reason_is_not_success(self):
  p=fixture(self.d);t=json.loads(p['terminal'].read_text());del t['abort_reason'];p['terminal'].write_text(json.dumps(t));v=self.parse(p);self.assertFalse(v['complete_diagnostic']);self.assertIn('terminal missing abort_reason',v['issues'])
 def test_nonfinite_fail_closed(self):
  p=fixture(self.d);p['stdout'].write_text(p['stdout'].read_text().replace('"fullslot_real_sine_max_complex": 0.0002','"fullslot_real_sine_max_complex": 1e999',1));v=self.parse(p);self.assertFalse(v['complete_diagnostic']);self.assertTrue(v['failures']);parse.write_outputs(self.d,v)
 def test_old_conditioned_null_is_expected_and_missing_is_failure(self):
  p=fixture(self.d);v=self.parse(p);self.assertEqual(v['status'],'complete-pass');s=json.loads(p['summary'].read_text());del s['cases'][0]['old_lane0']['post_Gate_refresh_decodedUp_conditioned'];p['summary'].write_text(json.dumps(s));p['stdout'].write_text(p['stdout'].read_text().replace('"post_Gate_refresh_decodedUp_conditioned": null, ','',1));v=self.parse(p);self.assertFalse(v['complete_diagnostic']);self.assertTrue(any('old_lane0 missing conditioned field' in x for x in v['issues']))
 def test_old_conditioned_nonnull_fails(self):
  p=fixture(self.d);txt=p['stdout'].read_text().replace('"post_Gate_refresh_decodedUp_conditioned": null','"post_Gate_refresh_decodedUp_conditioned": 0.1',1);p['stdout'].write_text(txt);v=self.parse(p);self.assertTrue(any('conditioned must be null' in x for x in v['issues']));self.assertFalse(v['complete_diagnostic'])
 def test_partial_rc2_not_complete_negative(self):
  p=fixture(self.d,'full');p['stdout'].write_text('\n'.join(p['stdout'].read_text().splitlines()[:-2])+'\n');v=self.parse(p);self.assertFalse(v['complete_diagnostic']);self.assertFalse(v['complete_finite_numeric_negative_rc2'])
 def test_missing_counter_is_not_summary_inference(self):
  p=fixture(self.d);p['stdout'].write_text('\n'.join(x for x in p['stdout'].read_text().splitlines() if 'physical_boot_calls=4 pairs=4' not in x)+'\n');v=self.parse(p);self.assertFalse(v['complete_diagnostic']);self.assertEqual(v['observed_counts']['projection_pair_counter_events'],15)
 def test_counter_pair_mismatch(self):
  p=fixture(self.d);p['stdout'].write_text(p['stdout'].read_text().replace('physical_boot_calls=4 pairs=4','physical_boot_calls=4 pairs=5',1));v=self.parse(p);self.assertFalse(v['complete_diagnostic']);self.assertIn('actual projection pair counter mismatch',v['issues'])
 def test_refresh_ordinal_mismatch(self):
  p=fixture(self.d);p['stdout'].write_text(p['stdout'].read_text().replace('"ordinal": 0, "lane": 1','"ordinal": 0, "lane": 0',1));v=self.parse(p);self.assertFalse(v['complete_diagnostic']);self.assertIn('case refresh ordinal/lane order mismatch',v['issues'])
 def test_guard_abnormal_child_not_numeric(self):
  p=fixture(self.d);t=json.loads(p['terminal'].read_text());t['returncode']=-9;p['terminal'].write_text(json.dumps(t));v=self.parse(p);self.assertEqual(v['status'],'structural-or-app-failure');self.assertFalse(v['complete_finite_numeric_negative_rc2'])
 def test_guard_abort_not_numeric(self):
  p=fixture(self.d);t=json.loads(p['terminal'].read_text());t.update(returncode=-15,abort_reason='foreign GPU');p['terminal'].write_text(json.dumps(t));v=self.parse(p);self.assertEqual(v['status'],'guard-abort');self.assertFalse(v['complete_finite_numeric_negative_rc2'])
 def test_live_partial_files_are_pending(self):
  p=fixture(self.d);p['terminal'].unlink();p['summary'].write_text('{');p['stdout'].write_text('PUBLIC_SILU_PAIR_SETUP {');p['gpu'].write_text('{');v=self.parse(p);self.assertEqual(v['status'],'incomplete',v['issues']);self.assertEqual(v['capture_states']['summary'],'partial-or-invalid-while-live');self.assertTrue(v['capture_states']['stdout_partial_line'])
 def test_no_capture_does_not_forge_counts(self):
  v=self.parse({});self.assertEqual(v['status'],'incomplete');self.assertIsNone(v['observed_counts']['last_physical_Boot_counter']);self.assertIsNone(v['observed_counts']['internal_EvalMod_calls_derived_not_independently_counted'])

class Node2SpecificTest(unittest.TestCase):
 setUp=ParserTest.setUp
 tearDown=ParserTest.tearDown
 parse=ParserTest.parse
 # Additional node2 cases; the original 18 tests above retain their gates.
 def test_structural_prefix_16_not_complete_32(self):
  p=fixture(self.d);seq=p['stdout'].read_text().splitlines();first=next(i for i,x in enumerate(seq) if x.startswith('PUBLIC_SILU_PAIR_CASE '));p['stdout'].write_text('\n'.join(seq[:first])+ '\nPUBLIC_SILU_PAIR_FAILED projection actual N/Q/id/scale/components/device mismatch\n');p['summary'].write_text('');t=json.loads(p['terminal'].read_text());t.update(returncode=1,classification='structural_or_resource_failure');p['terminal'].write_text(json.dumps(t));p['guard_launch'].write_text(json.dumps(t));v=self.parse(p)
  self.assertEqual(v['status'],'structural-or-app-failure');self.assertEqual(v['child_returncode'],1);self.assertIsNone(v['numerical_gates_passed']);self.assertEqual(v['observed_counts']['physical_Boot_counter_events'],16);self.assertEqual(v['observed_counts']['projection_pair_counter_events'],8);self.assertEqual(v['observed_counts']['case_rows'],0);self.assertEqual(v['observed_refresh_checks']['rows'],8);self.assertTrue(v['observed_refresh_checks']['all_observed_fixed_refresh_gates_pass']);self.assertIsNone(v['final_validation']['OFF_ON_same_plan_metadata_RNS_exact']);self.assertIsNone(v['timings']['incomplete_case_diagnostic_seconds']);self.assertEqual(v['issues'],[])
 def test_orphan_refresh_metadata_still_checked(self):
  p=fixture(self.d);seq=p['stdout'].read_text().splitlines();first=next(i for i,x in enumerate(seq) if x.startswith('PUBLIC_SILU_PAIR_CASE '));p['stdout'].write_text('\n'.join(seq[:first]).replace('"immutable": true','"immutable": false',1)+'\n');v=self.parse(p);self.assertTrue(any('observed refresh 0 immutable' in x for x in v['issues']));self.assertFalse(v['complete_diagnostic'])
 def test_node2_wrong_authorization(self):
  p=fixture(self.d);t=json.loads(p['terminal'].read_text());t['authorization_sha256']='0'*64;p['terminal'].write_text(json.dumps(t));v=self.parse(p);self.assertTrue(any('authorization/guard' in x for x in v['issues']));self.assertFalse(v['complete_diagnostic'])
 def test_node2_old_terminal_schema_rejected(self):
  p=fixture(self.d);t=json.loads(p['terminal'].read_text());t['requested_signal']=None;del t['requested_signals'];p['terminal'].write_text(json.dumps(t));v=self.parse(p);self.assertIn('terminal missing requested_signals',v['issues']);self.assertFalse(v['complete_diagnostic'])
 def test_node2_classification_mismatch(self):
  p=fixture(self.d);t=json.loads(p['terminal'].read_text());t['classification']='numerical_negative';p['terminal'].write_text(json.dumps(t));v=self.parse(p);self.assertIn('guard classification mismatch',v['issues']);self.assertFalse(v['complete_diagnostic'])
 def test_node2_terminal_registered_map_mismatch(self):
  p=fixture(self.d);t=json.loads(p['terminal'].read_text());t['registered']['nvml_host_pid']=101;p['terminal'].write_text(json.dumps(t));v=self.parse(p);self.assertIn('terminal vs PID map registration mismatch',v['issues']);self.assertFalse(v['complete_diagnostic'])
 def test_postflight_foreign_no_release_claim(self):
  p=fixture(self.d);t=json.loads(p['postflight'].read_text());t['compute_apps']['rows']=[[UUID,'101','x','10']];p['postflight'].write_text(json.dumps(t));v=self.parse(p);self.assertFalse(v['postflight_checks']['target_gpu_zero_MiB_no_compute_PID']);self.assertFalse(v['complete_diagnostic'])
 def test_postflight_guard_identity_absent_field_not_absent_process(self):
  p=fixture(self.d);t=json.loads(p['postflight'].read_text());del t['guard_identity_once'];p['postflight'].write_text(json.dumps(t));v=self.parse(p);self.assertFalse(v['postflight_checks']['expected_guard_absent']);self.assertFalse(v['complete_diagnostic'])
 def test_complex_norm_not_independent_re_im_bounds(self):
  p=fixture(self.d);s=json.loads(p['summary'].read_text());e=s['cases'][0]['paired_lane0'];e['fullslot_max_complex_source']=(2*.0008**2)**.5;e['max_imaginary']=.0008;e['fullslot_source_pass']=False;s['cases'][0]['all_fixed_gates_pass']=False;s['all_numerical_gates_passed']=False;p['summary'].write_text(json.dumps(s));seq=p['stdout'].read_text().splitlines();seq=[('PUBLIC_SILU_PAIR_CASE '+json.dumps(s['cases'][0])) if x.startswith('PUBLIC_SILU_PAIR_CASE ') and '"name": "a_b"' in x else x for x in seq];seq[-1]=seq[-1].replace('numerical_pass=1','numerical_pass=0');p['stdout'].write_text('\n'.join(seq)+'\n');t=json.loads(p['terminal'].read_text());t.update(returncode=2,classification='numerical_negative');p['terminal'].write_text(json.dumps(t));p['guard_launch'].write_text(json.dumps(t));v=self.parse(p);self.assertEqual(v['status'],'complete-numerical-negative',v['issues']);self.assertFalse(v['numerical_gates_passed'])

if __name__=='__main__':unittest.main(verbosity=2)
