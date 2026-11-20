"""New A5r source metadata and fake-supervision fixtures only."""
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.experiments import a5r_resnet_rule63_shared_optimized_compile as own
from pt_agent.experiments import cold_rule63_whole_model as base
from pt_agent.experiments import dacapo_compile_continuation as resources


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(data))


def clean(start=100, end=200):
    return dict(started_ns=start, ended_ns=end, exit_code=0, timed_out=False, signal=None,
                remaining_pids=[], signals_sent=[], cleanup_unknown=False)


class A5rResNet(unittest.TestCase):
    def fixture(self, root, *, archived=True):
        source = root/own.SOURCE_ID; archive_case = root/own.ARCHIVE_ID
        profile = dict(polynomialDegree=65536, rescalingFactor=43, levelLowerBound=1, levelUpperBound=8,
            bootstrapLevelLowerBound=1, bootstrapLevelUpperBound=5, runtime='HEAAN-HEVM')
        recipe = ['hecate-opt', 'old.mlir', '--dacapo', '--boot-placement=depth-dp', '-o', 'old.lowered.mlir']
        profile_path=root/'profile.json'; template=root/'template.json'
        write(profile_path, profile); write(template, recipe)
        write(source/'source/compiler-profile.json', profile)
        write(source/'source/compile-metadata-template.json', recipe)
        runtime=source/'work/emit/lowered.fixture.runtime-plan.json';write(runtime, dict(fixture=True))
        count=dict(counts=dict(boot=0, mul_cc=2), ops_total=2, distinct_consts=1)
        phase={p:dict(status='ok')for p in base.PHASES}
        phase['generation'].update(rule_options=own.shared.OPTIONS, raw_Count=dict(stats=count),
            signature=dict(passed=True), training_math=dict(samples=3,checks=[dict(passed=True)]*3))
        phase['emit']['emit_stats']=count
        phase['trace']['trace_bytes']=64
        phase['compile'].update(native_process=clean(), compiled_bootstraps=230, runtime_plan=str(runtime))
        result=dict(round=base.ROUND, mode=base.MODE, run_id=source.name, model=own.MODEL, placement='depth',
            status='ok', git_rev='fixture', monitor_stopped=True, guard_error=None, monitor_error=None,
            expected_bootstraps_match=True, attempted=dict.fromkeys(base.PHASES,1), completed=dict.fromkeys(base.PHASES,1),
            phases=phase, processes=[dict(phase=p, observation=clean(10,250)) for p in base.PHASES],
            source=dict(compiler=str(root/'hecate-opt')), runtime_plan=str(runtime))
        write(source/'result.json',result)
        write(source/'completion.json',dict(round=base.ROUND,status='ok',started_ns=1,ended_ns=300,
            registry_persisted=True,required_artifacts_persisted=True))
        raw=source/'work/emit/candidate_trace.mlir'; gz=raw.with_name(raw.name+'.gz')
        if archived:
            gz.write_text('metadata-only fixture, never decoded')
            ar=dict(round='A-GROUP-COLD-CLOSED-RESOURCE-ARCHIVE',status='ok',source_case=str(source),
                scientific_status='ok',resource_status='archived_or_kept',files=[dict(source=str(raw),
                    archive=str(gz),status='compressed',raw_bytes=64,compressed_bytes=gz.stat().st_size)])
            write(archive_case/'result.json',ar)
            write(archive_case/'completion.json',dict(round=ar['round'],status='ok',registry_persisted=True,
                required_artifacts_persisted=True))
        else:raw.write_text('metadata-only unarchived fixture')
        def ledger(case, round_, model, mode):
            return dict(run_id=case.name, round=round_,model=model,mode=mode,git_rev='fixture',
                metrics=dict(status='ok'),artifacts_dir=str(case),ts='fixture')
        rows=[ledger(source,base.ROUND,own.MODEL,base.MODE)]
        if archived:rows.append(ledger(archive_case,ar['round'],source.name,'resource-only-reversible-gzip'))
        (root/'registry.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in rows))
        args=own.arguments(['--source-case',str(source),'--out',str(root/'fresh-a5r'),
            '--paper-runs',str(root/'paper.jsonl'),'--host-pid-namespace',os.readlink('/proc/self/ns/pid'),
            '--profile',str(profile_path),'--metadata-template',str(template),'--seconds','9600'])
        return args,result,dict(compiler=str(root/'hecate-opt'))

    def test_exact_archived_or_raw_source_proof_without_payload_read(self):
        for archived in (True,False):
            with self.subTest(archived=archived), tempfile.TemporaryDirectory()as tmp,\
                    patch.object(paths,'RESULTS_DIR',Path(tmp)),\
                    patch.object(own.shared.closed_source,'producer_births',return_value=9):
                args,result,current=self.fixture(Path(tmp),archived=archived)
                original_open=Path.open
                def open_metadata(path,*a,**kw):
                    self.assertNotEqual(path.suffix,'.gz','Source gate must not read gzip payload')
                    return original_open(path,*a,**kw)
                with patch.object(base,'admission',return_value=current),patch.object(Path,'open',open_metadata):
                    proof=own.source_gate(args)
                self.assertEqual(proof['input_kind'],'archived-input-preparation'if archived else'raw-symlink')
                self.assertEqual(proof['shared_generation_prefix_ns'],99)
                self.assertEqual(proof['source_payload_reads'],0)
                self.assertTrue(proof['source_independent_math_passed'])
                result['phases']['generation']['rule_options']=dict(deg=31)
                write(args.source_case/'result.json',result)
                with patch.object(base,'admission',return_value=current),self.assertRaisesRegex(ValueError,'Fixed Rule63'):
                    own.source_gate(args)

    def test_pure_worker_command_targets_new_module_and_explicit_budget(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(paths,'RESULTS_DIR',Path(tmp)):
            args,_,_=self.fixture(Path(tmp));args.out.mkdir()
            write(args.out/'source/shared-generation-source.json',dict(input_kind='archived-input-preparation',
                input_source=str(Path(tmp)/'fixture.gz')))
            for phase,budget in (('restore',1800),('compile',7200)):
                command,timeout=own.phase_command(args,phase,9600)
                separator=command.index('--');inner=command[separator+1:]
                self.assertEqual(inner[:3],['-B','-m',own.MODULE])
                self.assertEqual(inner[inner.index('--native-seconds')+1],str(budget))
                selected=own.arguments(inner[3:])
                self.assertEqual(selected.model,own.MODEL);self.assertEqual(selected.worker_phase,phase)
                self.assertEqual(timeout,budget+120)
                self.assertEqual(selected.emit_dir,args.out/'input')

    def test_fake_execute_success_and_source_rejection_durable_rows(self):
        for rejected in (False,True):
            with self.subTest(rejected=rejected), tempfile.TemporaryDirectory()as tmp,\
                    patch.object(paths,'RESULTS_DIR',Path(tmp)),\
                    patch.object(own.shared.closed_source,'producer_births',return_value=9),\
                    patch.object(resources,'available_ram',return_value=100*base.GIB):
                args,_,current=self.fixture(Path(tmp));calls=[]
                def native_argv(_args,proof):
                    return ['synthetic-native',str(_args.out/'input/candidate_trace.mlir'),'-o',str(_args.out/'input/lowered.mlir')]
                def fake_process(command,**kwargs):
                    phase=command[command.index('--worker-phase')+1];calls.append(phase)
                    start=time.perf_counter_ns()
                    if phase=='restore':
                        (args.out/'input/candidate_trace.mlir').write_text('new-case fake recovered fixture')
                        value=dict(status='ok')
                    else:
                        proof=own.read(args.out/'source/shared-generation-source.json')
                        write(args.out/'work/compile/native-argv.json',native_argv(args,proof))
                        value=dict(status='ok',native_process=clean(start,time.perf_counter_ns()),
                            compiled_bootstraps=123,execution_steps=456,runtime_plan='synthetic',
                            prediction=None,predicted_latency_us=None)
                    write(args.out/'work'/phase/'phase_result.json',value)
                    return clean(start,time.perf_counter_ns())
                def admit(*a,**kw):
                    if rejected:raise ValueError('synthetic source unavailable')
                    return current
                terminal=own.private(own.bridge.terminal_source_observation,expected_native_argv=native_argv)
                execution=own.private(own.execute,prepare_source=own.prepare_source)
                with patch.object(base,'admission',side_effect=admit),patch.object(base,'run_process',side_effect=fake_process),\
                        patch.object(own,'expected_native_argv',side_effect=native_argv),\
                        patch.object(own.bridge,'terminal_source_observation',terminal):
                    # execute globals are private; replace its existing terminal hook explicitly.
                    execution=own.private(execution,terminal_source_observation=terminal)
                    result=execution(args)
                self.assertEqual(calls,[]if rejected else['restore','compile'])
                self.assertEqual(result['status'],'error'if rejected else'ok')
                self.assertEqual(result['round'],own.ROUND)
                self.assertIn('ResNet',result['actual_clock_scope'])
                completion=own.read(args.out/'completion.json');self.assertTrue(completion['registry_persisted'])
                row=own.read(args.out/'paper-record.json')
                self.assertEqual(row['model'],'resnet20');self.assertEqual(row['task'],'A5r')
                self.assertEqual(row['compile_attempts'],0 if rejected else 1)
                self.assertEqual(row['status'],'driver_error'if rejected else'ok')
                self.assertEqual(row['cold'],False);self.assertIsNone(row['full_inference_prediction_s'])
                if not rejected:
                    self.assertEqual(row['derived_e2e_s'],row['shared_generation_prefix_s']+row['compile_s'])
                    self.assertAlmostEqual(sum(v for v in row['stages_s'].values()if v is not None),
                        completion['complete_case_elapsed_seconds'],places=12)
                    self.assertTrue((args.source_case/'work/emit/candidate_trace.mlir.gz').is_file())
                    self.assertFalse((args.source_case/'work/emit/candidate_trace.mlir').exists())
                    self.assertFalse((args.out/'work/emit/candidate_trace.mlir').exists())
                self.assertEqual(len((Path(tmp)/'paper.jsonl').read_text().splitlines()),1)


if __name__=='__main__':unittest.main()
