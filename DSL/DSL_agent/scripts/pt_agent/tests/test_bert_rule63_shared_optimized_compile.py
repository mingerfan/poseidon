"""Synthetic input-format and fake-compile fixtures; original archives unopened."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

from pt_agent import paths, registry, toolchain
from pt_agent.experiments import bert_rule63_shared_optimized_compile as own
from pt_agent.experiments import dacapo_compile_continuation as resources
from pt_agent.experiments.bounded_gzip_input import CHUNK, restore_raw_member


def member(text, *, optional=False, level=6, footer=b'\x00' * 8):
    encoder = zlib.compressobj(level=level, wbits=-15)
    body = encoder.compress(text) + encoder.flush()
    header = b'\x1f\x8b\x08' + (b'\x1e' if optional else b'\x00') + b'\x00' * 6
    if optional: header += b'\x04\x00testname\x00comment\x00\xff\xfe'
    return header + body + footer


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


def observation(start=100, end=200, **changes):
    value = dict(exit_code=0, timed_out=False, signal=None, remaining_pids=[], signals_sent=[],
                 started_ns=start, ended_ns=end)
    value.update(changes); return value


class BoundedInput(unittest.TestCase):
    def decode(self, root, encoded, name='decoded', limit=4 * CHUNK, guard=None):
        source=root/(name+'.gz'); source.write_bytes(encoded); target=root/name
        result=restore_raw_member(source,target,maximum_output_bytes=limit,guard=guard or (lambda n:None))
        return target,result

    def test_standard_and_optional_headers_skip_FHCRC_and_footer_values(self):
        # These are tiny synthetic format fixtures, not scientific readback checks.
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for i,(optional,footer) in enumerate(((False,b'\x00'*8),(False,b'\xff'*8),(True,b'\xaa'*8))):
                target,result=self.decode(root,member(b'fixture input',optional=optional,footer=footer),str(i))
                self.assertEqual(target.read_text(),'fixture input')
                self.assertFalse(result['footer_parsed'])
                self.assertEqual(result['CRC_checks'],0); self.assertEqual(result['FHCRC_checks'],0)
                self.assertEqual(result['decoded_size_identity_comparisons'],0)

    def test_compressed_and_decoded_streaming_use_multiple_bounded_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            seen=[]; text=b'x'*(2*CHUNK+19)
            target,result=self.decode(Path(temporary),member(text,level=0),guard=seen.append)
            self.assertEqual(result['decoded_bytes_written'],len(text))
            self.assertGreaterEqual(result['compressed_blocks'],3)
            self.assertGreaterEqual(result['decoded_blocks'],3)
            self.assertLessEqual(max(seen),CHUNK)
            self.assertEqual(target.stat().st_size,len(text))

    def test_truncated_header_and_deflate_preserve_partial_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); encoder=zlib.compressobj(wbits=-15)
            body=encoder.compress(b'fixture data')+encoder.flush()
            for i,encoded in enumerate((b'\x1f\x8b\x08\x08'+b'\x00'*6+b'unterminated',
                                       b'\x1f\x8b\x08\x00'+b'\x00'*6+body[:-1])):
                source=root/(str(i)+'.gz');source.write_bytes(encoded);target=root/str(i)
                with self.assertRaises(EOFError):
                    restore_raw_member(source,target,maximum_output_bytes=CHUNK,guard=lambda n:None)
                self.assertTrue(target.exists())

    def test_output_file_and_external_free_disk_guard_keep_partial_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); encoded=member(b'y'*(3*CHUNK),level=0)
            source=root/'fixture.gz';source.write_bytes(encoded)
            with self.assertRaisesRegex(RuntimeError,'output-file allowance'):
                restore_raw_member(source,root/'file-limit',maximum_output_bytes=CHUNK+100,guard=lambda n:None)
            self.assertGreater((root/'file-limit').stat().st_size,0)
            self.assertLessEqual((root/'file-limit').stat().st_size,CHUNK+100)
            writes=[]
            def guard(additional):
                if additional:
                    if writes: raise RuntimeError('resource_limit: synthetic free disk guard')
                    writes.append(additional)
            with self.assertRaisesRegex(RuntimeError,'free disk guard'):
                restore_raw_member(source,root/'disk-limit',maximum_output_bytes=4*CHUNK,guard=guard)
            self.assertEqual((root/'disk-limit').stat().st_size,writes[0])


class BertSharedCompile(unittest.TestCase):
    def fixture(self,root,*,archived=True):
        profile=root/'compiler-config.json'; template=root/'compile_argv.json'; binary=root/'hecate-opt'
        binary.write_text('synthetic existing executable')
        write(profile,dict(polynomialDegree=65536,rescalingFactor=43,levelLowerBound=1,levelUpperBound=8,
            bootstrapLevelLowerBound=1,bootstrapLevelUpperBound=5,runtime='HEAAN-HEVM',latencyTable={}))
        write(template,['{hecate_opt}','{input}','--pars','-o','{output_dir}/lowered.mlir',
            '--compiler-config={package}/config/compiler-config.json','--runtime-plan-context-id=qwen-bootstrap48',
            '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5','--waterline=43',
            '--runtime-plan-operator-spec-sha256=sha256:'+'0'*64])
        case=root/own.SOURCE_ID;emit=case/'work/emit';emit.mkdir(parents=True)
        raw=emit/'candidate_trace.mlir'; archive=raw.with_name(raw.name+'.gz')
        if archived:archive.write_bytes(member(b'synthetic compiler input'))
        else:raw.write_text('synthetic compiler input')
        write(emit/'lowered.fixture.runtime-plan.json',dict(synthetic=True))
        write(case/'compiler-config.json',own.read(profile));write(case/'recorded-compile-template.json',own.read(template))
        phases={p:dict(status='ok')for p in own.source_driver.PHASES}
        phases['prepare'].update(activation_degree_changes=[dict(fn=fn,original_degree=127,degree=63)
            for fn in('gelu','gelu','tanh')],raw_count=dict(counts=dict(boot=0,mul_cc=2),ops_total=9,distinct_consts=3),signature=dict(ok=True))
        phases['emit'].update(emit_stats=dict(phases['prepare']['raw_count']),signature=dict(ok=True))
        phases['trace'].update(process=observation(),mlir=dict(bytes=64))
        phases['compile'].update(process=observation(),compiled_bootstraps=12713,
            runtime_plan=str(emit/'lowered.fixture.runtime-plan.json'))
        result=dict(round=own.source_driver.ROUND,model=own.source_driver.MODEL,mode=own.source_driver.MODE,
            status='ok',compile_accepted=True,monitor_stopped=True,compiled_bootstraps=12713,
            attempted=dict.fromkeys(own.source_driver.PHASES,1),completed=dict.fromkeys(own.source_driver.PHASES,1),
            phases=phases,processes=[dict(phase=p,observation=observation())for p in own.source_driver.PHASES],
            runtime_plan=phases['compile']['runtime_plan'],source=dict(hecate_opt=str(binary)),
            **dict.fromkeys(own.source_driver.ZERO,0))
        write(case/'result.json',result);write(case/'completion.json',dict(round=result['round'],status='ok',registry_persisted=True,required_artifacts_persisted=True))
        def ledger(directory,round_label,model,mode):
            return dict(run_id=directory.name,round=round_label,model=model,mode=mode,git_rev='fixture',
                metrics=dict(status='ok'),artifacts_dir=str(directory),ts='fixture')
        records=[ledger(case,result['round'],result['model'],result['mode'])]
        if archived:
            ac=root/own.ARCHIVE_ID
            ar=dict(round='A1-CLOSED-RAW-ARCHIVE',status='ok',source_case=str(case),process=observation(),
                archive=dict(source=str(raw),archive=str(archive),status='compressed',raw_bytes=64,compressed_bytes=archive.stat().st_size))
            write(ac/'result.json',ar);write(ac/'completion.json',dict(round=ar['round'],status='ok',registry_persisted=True,required_artifacts_persisted=True))
            records.append(ledger(ac,ar['round'],'bert_tiny','closed-raw-MLIR-reversible-compression'))
        (root/'registry.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in records))
        args=own.arguments(['--source-case',str(case),'--out',str(root/'new-compile'),'--paper-runs',str(root/'paper.jsonl'),
            '--host-pid-namespace',os.readlink('/proc/self/ns/pid'),'--profile',str(profile),'--metadata-template',str(template),
            '--diagnostic-elide-final-ir'])
        return args,result,binary

    def test_source_gates_raw_or_archive_without_claiming_unmeasured_math(self):
        for archived in(False,True):
            with self.subTest(archived=archived),tempfile.TemporaryDirectory()as temporary,patch.object(paths,'RESULTS_DIR',Path(temporary)):
                root=Path(temporary);args,source,binary=self.fixture(root,archived=archived)
                with patch.object(toolchain,'host_paths',return_value={'hecate_opt':str(binary)}),\
                        patch.object(own.shared.closed_source,'producer_births',return_value=5):
                    proof=own.source_gate(args)
                    self.assertEqual(proof['input_kind'],'archived-input-preparation'if archived else'raw-symlink')
                    self.assertIsNone(proof['independent_math_passed']);self.assertEqual(proof['source_payload_reads'],0)
                    with patch.object(own.shutil,'disk_usage',return_value=type('Disk',(),{'free':9*own.supervisor.GIB})()):
                        with self.assertRaisesRegex(ValueError,'resource_limit'):own.source_gate(args)
                    source['phases']['emit']['emit_stats']=dict(counts=dict(boot=1),ops_total=9,distinct_consts=3)
                    write(args.source_case/'result.json',source)
                    with self.assertRaisesRegex(ValueError,'Count/Emit'):own.source_gate(args)

    def test_pure_worker_arguments_restore_and_native_have_separate_budgets(self):
        with tempfile.TemporaryDirectory()as temporary:
            args,_,_=self.fixture(Path(temporary));(args.out/'source').mkdir(parents=True)
            write(args.out/'source/shared-generation-source.json',dict(input_kind='archived-input-preparation',input_source='/fixture.gz'))
            for phase,cap in(('restore',1800),('compile',7200)):
                command,seconds=own.phase_command(args,phase,9500)
                self.assertEqual(command.count('--python'),1)
                self.assertEqual(command[command.index('--native-seconds')+1],str(cap))
                self.assertEqual(seconds,cap+120)
                worker=own.arguments(command[command.index('--')+1:][3:])
                self.assertEqual(worker.worker_phase,phase)
                self.assertEqual(worker.archive,Path('/fixture.gz'))
                self.assertNotIn('--source-case',command)

    def test_fake_restore_and_native_or_source_rejection_preserve_closure(self):
        frozen=(own.supervisor.ROUND,own.supervisor.MODE,own.supervisor.PHASES,own.supervisor.execute.__code__)
        for reject in(False,True):
            with self.subTest(reject=reject),tempfile.TemporaryDirectory()as temporary,patch.object(paths,'RESULTS_DIR',Path(temporary)):
                root=Path(temporary);args,source,binary=self.fixture(root)
                if reject:source['status']='error';write(args.source_case/'result.json',source)
                calls=[]
                def process(command,**kwargs):
                    phase=command[command.index('--worker-phase')+1];calls.append(phase)
                    self.assertEqual(kwargs['mem_gib'],24);self.assertEqual(kwargs['fsize_gib'],16)
                    value=dict(status='ok',error=None)
                    if phase=='restore':
                        (args.out/'input/candidate_trace.mlir').write_text('synthetic fresh input')
                        value['restore']=dict(decoded_bytes_written=21,CRC_checks=0)
                    else:
                        value.update(native_process=observation(300,500),compiled_bootstraps=1234,execution_steps=4567,
                            runtime_plan='synthetic-optimized-plan',execution_ops=dict(boot=1234),
                            prediction=dict(table_known_sum_us=2_500_000,missing=[dict(op='relinearize')]))
                        write(args.out/'work/compile/native-argv.json',own.read(args.out/'source/expected-native-argv.json'))
                    write(args.out/'work'/phase/'phase_result.json',value)
                    return observation(250,550)
                with patch.object(toolchain,'host_paths',return_value={'hecate_opt':str(binary)}),\
                        patch.object(own.shared.closed_source,'producer_births',return_value=5),\
                        patch.object(own.supervisor,'run_process',side_effect=process),\
                        patch.object(registry,'git_rev',return_value='fixture'),\
                        patch.object(resources,'available_ram',return_value=20*own.supervisor.GIB):
                    result=own.execute(args)
                self.assertEqual(calls,[]if reject else['restore','compile'])
                self.assertEqual(result['attempted'],{}if reject else dict(restore=1,compile=1))
                self.assertEqual(result['status'],'error'if reject else'ok')
                self.assertEqual(own.read(args.out/'completion.json')['status'],result['status'])
                row=own.read(args.out/'paper-record.json')
                self.assertEqual(row['status'],'driver_error'if reject else'ok')
                self.assertIsNone(row['generation_e2e_s']);self.assertIsNone(row['independent_math_passed'])
                self.assertEqual(row['pred_latency_s'],None if reject else 2.5)
                self.assertAlmostEqual(sum(v for v in row['stages_s'].values()if v is not None),row['actual_compile_only_case_seconds'])
                records=[json.loads(line)for line in(root/'registry.jsonl').read_text().splitlines()]
                self.assertEqual(len(records),3);self.assertTrue(all(len(r)==8 for r in records))
                self.assertTrue((args.source_case/'work/emit/candidate_trace.mlir.gz').is_file())
                self.assertFalse((args.source_case/'work/emit/candidate_trace.mlir').exists())
                if not reject:
                    self.assertTrue(result['source_input_stat_preserved']);self.assertTrue(result['native_argv_verified'])
                    self.assertFalse((args.out/'input/candidate_trace.mlir').is_symlink())
                    self.assertFalse((args.out/'work/emit/candidate_trace.mlir').exists())
        self.assertEqual(frozen,(own.supervisor.ROUND,own.supervisor.MODE,own.supervisor.PHASES,own.supervisor.execute.__code__))


if __name__=='__main__':unittest.main()
