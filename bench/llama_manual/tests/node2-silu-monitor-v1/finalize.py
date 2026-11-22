#!/usr/bin/env python3
"""Local-only public aggregate provenance/report builder. No connection or app launch."""
import hashlib
import json
import pathlib
import re

HERE=pathlib.Path(__file__).resolve().parent
WORKTREE=HERE.parents[3]
ALIGN=WORKTREE.parent/'llama-cerium-align'
AUTH=ALIGN/'experiments/20261116-manual-llama/node2-v1/gpu-public-silu-v1/authorization.json'
LAUNCH=AUTH.parent/'launch.stdout.json'
BASE=ALIGN/'experiments/20261115-cerium-manual-align/a800-silu-pair-monitor-v1'

def load(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

v=load(HERE/'PARSED.json')
a=load(AUTH)
launch=load(LAUNCH)
c=load(HERE/'command.json')
t=load(HERE/'terminal.json')
assert sha(AUTH)=='ea218a2d67a0e04f5f439286b3784a2c915a7bbbb0e6682bbd68549929bb0d13'
checks={'local_authorization_sha_matches':True,
        'captured_command_matches_authorization':c['command']==a['command'],
        'captured_eight_lease_identities_match_authorization':c['owned_shared_leases']==a['leases'],
        'captured_environment_matches_authorization':all(c['env'].get(k)==x for k,x in a['environment'].items()),
        'captured_gpu_matches_authorization':c['initial_gpu'][0]==a['gpu_uuid'],
        'guard_launch_identity_matches':launch['guard_pid']==c['guard_pid']==95575 and launch['pgid']==95575 and launch['start_ticks']=='1373031093',
        'guard_final_print_matches_terminal':load(HERE/'guard-launch.log')==t}
assert all(checks.values()),checks
sources=['bench/llama_manual/tests/public_paired_silu.cpp','bench/llama_manual/tests/public_paired_projections.cpp',
         'bench/llama_manual/tests/source_silu_cpu.hpp','bench/llama_manual/source_manual_pair_silu.hpp',
         'bench/llama_manual/source_manual_boot.hpp','bench/llama_manual/source_accuracy.hpp',
         'bench/llama_manual/source_ops.hpp','bench/llama_manual/tools/run_node2_guard.py']
provenance={'framework_base_from_Main_briefing':'319d811959949e1ea9250d2d48dec8362e6d7729',
            'source_generation':'latest framework plus four additive core extensions; promoted source_manual_boot/silu/pair controls, not historical GPU generation',
            'source_generation_scope':'Main-provided provenance and frozen-root assumption; monitor did not fetch sources or independently rehash remote binaries',
            'authorization_path_local':str(AUTH),'authorization_sha256':sha(AUTH),'checks':checks,
            'executable_sha256_authorized':a['artifacts']['build/bin/llama31_public_paired_silu'],
            'backend_archive_sha256_authorized':a['artifacts']['build/libposeidon_gpu.a'],
            'cpu_shared_sha256_authorized':a['artifacts']['build/libposeidon_shared.so'],
            'remote_source_inventory_sha256_authorized':a['artifacts']['all-source-sha256.json'],
            'guard_artifact_hash_checks_scope':'guard performs pinned-artifact and source-inventory checks before admission; no monitor-side remote rehash',
            'reference_parser':{'path':str(BASE/'parse.py'),'sha256':sha(BASE/'parse.py')},
            'reference_tests':{'path':str(BASE/'test_parse.py'),'sha256':sha(BASE/'test_parse.py')},
            'local_source_snapshot_sha256':{name:sha(WORKTREE/name) for name in sources},
            'cpu_helper_40_boot_marker_scope':'old primitive planned/CPU schema marker, not actual SiLU GPU work',
            'expected_full_control':{'cases':2,'physical_Boots':32,'projection_pairs':16,'refresh_rows':16,'internal_EvalMod_calls_derived_only':64},
            'source_oracle':'G/G/F/F/PQ source approximation, not true/HF SiLU; complex norm <=1e-3 unchanged',
            'domain':'public Gate +/-25, Up +/-8; 32768 complex slots, active38=9728; not full32 model qualification'}
save('PROVENANCE.json',provenance)
setup_line=next((x for x in (HERE/'stdout.log').read_text().splitlines() if x.startswith('SOURCE_SETUP stage=completed seconds=')),None)
source_setup_seconds=float(setup_line.rsplit('=',1)[1]) if setup_line else None
report={'status':v['status'],'complete_diagnostic':v['complete_diagnostic'],
        'numerical_gates_passed':v['numerical_gates_passed'],'child_returncode':t['returncode'],
        'outer_classification':t['classification'],'abort_reason':t['abort_reason'],
        'requested_signals':t['requested_signals'],'cleanup_error':t.get('cleanup_error'),
        'owned_process_may_still_be_live':t.get('owned_process_may_still_be_live',False),
        'start':t['start'],'end':t['end'],'failure':v['failures'],
        'failure_location_source_inference':{'file':'bench/llama_manual/tests/public_paired_silu.cpp:179-186',
            'stage':'after first-case 16 calls, final shape(runtime,*x,10,2^40), before OFF/ON exact-RNS and final source metrics',
            'limits':'generic shape predicate; failing output and individual N/Q/parms_id/scale/component/device subfield are not logged; do not infer them'},
        'observed_counts':v['observed_counts'],'observed_refresh_checks':v['observed_refresh_checks'],
        'final_validation':v['final_validation'],
        'timings':{**v['timings'],'SOURCE_SETUP_completed_seconds':source_setup_seconds,
                   'scope':'reported setup/guard walls only; no completed case diagnostic wall was printed; no clean online/GPU-busy/performance result; do not subtract to invent diagnostic time'},
        'telemetry':v['telemetry'],'postflight':v['postflight'],'postflight_checks':v['postflight_checks'],
        'pid_map':v['pid_map'],'remote_SUMMARY_state':v['capture_states']['summary'],
        'remote_SUMMARY_bytes':(HERE/'SUMMARY.json').stat().st_size,
        'model_qualification':False,'performance_valid':False,
        'source_vs_true_SiLU_max_complex_nongating':v['source_vs_true_SiLU_max_complex'],
        'source_vs_true_SiLU_scope':v['source_vs_true_SiLU_scope'],
        'outer_exit_status_scope':v['outer_exit_status_scope'],
        'parser_issues':v['issues'],'parser_tests':{'final_passed':27,'scope':'pure local synthetic aggregate tests; no GPU/remote execution'},
        'collection':load(HERE/'COLLECTION.json'),
        'operations':'one existing-master SSH, terminal present on first tiny check; one postflight; no remote writes/apps/retries/repair/signals/reconnect/peer contact',
        'evidence_scope':'public aggregate logs only; no private weights/keys/tensors/CT/PT/arrays; source-only frozen-root assumption; telemetry noncontinuous'}
save('MONITOR-REPORT.json',report)
# Hash every retained local artifact except this self-referential inventory.
save('FINAL-INVENTORY.json',{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)}
                             for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='FINAL-INVENTORY.json'})
print(json.dumps({'status':report['status'],'provenance_checks':checks,'output':'MONITOR-REPORT.json'}))
