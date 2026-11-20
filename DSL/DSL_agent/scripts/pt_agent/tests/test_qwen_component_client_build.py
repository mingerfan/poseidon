"""Own metadata/fake-build and concrete Go-source contract checks only.

No Go parser/compiler/test/run, model, Count, Emit, native tool or SDK is run.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_component_client_build as entry


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)if isinstance(value,dict)else value)


def observation():return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])


def function(source,name):
    """Bounded lexical function body extraction, not Go execution/parsing."""
    begin=re.search(r'(?m)^func\s*(?:\([^\n]+\)\s*)?'+re.escape(name)+r'\(',source)
    if not begin:raise AssertionError('Concrete Go function absent: '+name)
    start=source.index('{',begin.end());depth=1;i=start+1;quoted=None;escaped=False;comment=None
    while i<len(source)and depth:
        char=source[i];pair=source[i:i+2]
        if comment=='line':
            if char=='\n':comment=None
        elif comment=='block':
            if pair=='*/':comment=None;i+=1
        elif quoted:
            if escaped:escaped=False
            elif char=='\\'and quoted!='`':escaped=True
            elif char==quoted:quoted=None
        elif pair=='//':comment='line';i+=1
        elif pair=='/*':comment='block';i+=1
        elif char in ('"',"'",'`'):quoted=char
        elif char=='{':depth+=1
        elif char=='}':depth-=1
        i+=1
    if depth:raise AssertionError('Unclosed concrete Go function: '+name)
    return source[start+1:i-1]


class ComponentLibraryBuildChecks(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.results=self.root/'results';self.results.mkdir();self.prepared=self.root/'r31';self.sdk=self.root/'sdk'
        self.lattigo=self.root/'existing-lattigo';self.lattigo.mkdir()
        put(self.sdk/'VERSION','go1.25.13\ntime retained\n');put(self.sdk/'bin/go','existing fake executable only\n');(self.sdk/'bin/go').chmod(0o700)
        put(self.prepared/'result.json',dict(round='R31',phase='prepare',status='ok',lattigo_source=str(self.lattigo)))
        put(self.prepared/'work/module/go.mod','module pt-agent-r31-plan\n\ngo 1.25.0\n\nrequire github.com/tuneinsight/lattigo/v6 v6.0.0\n\nreplace github.com/tuneinsight/lattigo/v6 => '+str(self.lattigo)+'\n')
        put(self.prepared/'work/module/go.sum','opaque existing Go metadata, not compared\n')
        for name in ('module-cache','build-cache'):put(self.prepared/'work'/name/'fixture','tiny fake cache')
        self.args=SimpleNamespace(out=self.results/'new',prepared=self.prepared,go_sdk=self.sdk,registry=self.results/'registry.jsonl',run=True)
        self.calls=[];self.go=entry.GO_SOURCE.read_text()

    def run_fake(self,runner=None,gate=None):
        def build(argv,**kwargs):
            self.calls.append((argv,kwargs))
            if runner:return runner(argv,**kwargs)
            put(Path(argv[argv.index('-o')+1]),'synthetic archive for fake-process fixture only\n')
            return observation()
        with patch.object(entry.paths,'RESULTS_DIR',self.results),patch.object(entry,'run_process',side_effect=build), \
                patch.object(entry,'_available_ram_bytes',return_value=16*entry.GIB), \
                patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=100*entry.GIB)), \
                patch.object(entry,'source_gate',side_effect=gate or entry.source_gate):
            return entry.execute(self.args)

    def test_original_sdk_cache_and_module_admission_without_any_tool(self):
        source=entry.source_gate(self.args)
        self.assertEqual(source['go_version'],'1.25.13');self.assertEqual(source['lattigo_version'],'6.0.0')
        self.assertFalse(source['parameters_selected']);self.assertEqual(source['actual_component_artifacts_admitted'],0)
        self.assertEqual(source['lattigo_source'],str(self.lattigo));self.assertEqual(source['source_integrity_comparisons'],0)

    def test_failed_prepare_wrong_sdk_or_dependency_override_rejected(self):
        path=self.prepared/'result.json';record=entry.read(path);put(path,dict(record,status='error'))
        with self.assertRaises(ValueError):entry.source_gate(self.args)
        put(path,record);put(self.sdk/'VERSION','go1.26.0\n')
        with self.assertRaises(ValueError):entry.source_gate(self.args)
        put(self.sdk/'VERSION','go1.25.13\n')
        mod=self.prepared/'work/module/go.mod';mod.write_text(mod.read_text()+'toolchain go1.26.0\n')
        with self.assertRaises(ValueError):entry.source_gate(self.args)

    def test_constructor_or_executable_source_is_rejected_before_build(self):
        source=self.root/'bad.go';source.write_text(self.go+'\nfunc init(){NewParametersFromLiteral()}\n')
        with patch.object(entry,'GO_SOURCE',source):
            with self.assertRaises(ValueError):entry.source_gate(self.args)

    def test_full_fake_archive_build_exact_limits_and_explicit_unlinked_scope(self):
        result=self.run_fake()
        self.assertEqual(result['status'],'ok');self.assertEqual(result['attempted_builds'],1)
        self.assertEqual(result['completed_builds'],1);self.assertTrue(result['generic_library_build_passed'])
        self.assertTrue(result['generic_library_compile_accepted']);self.assertFalse(result['linked_native_program_verified'])
        self.assertFalse(result['full_qwen_compile_verified']);self.assertFalse(result['whole_generation_verified'])
        self.assertIsNone(result['actual_component_manifest']);self.assertEqual(result['actual_component_artifacts_admitted'],0)
        self.assertTrue(all(result[k]==0 for k in entry.ZERO));self.assertEqual(len(self.calls),1)
        argv,kwargs=self.calls[0]
        self.assertEqual(argv[1:5],['build','-p=1','-mod=readonly','-buildvcs=false'])
        self.assertIn('-buildmode=archive',argv);self.assertNotIn('run',argv);self.assertNotIn('test',argv)
        self.assertEqual(kwargs['mem_gib'],4);self.assertEqual(kwargs['fsize_gib'],8);self.assertLessEqual(kwargs['timeout'],600)
        row=entry.read(self.args.registry)
        self.assertEqual(set(row),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        self.assertFalse(row['metrics']['linked_native_program_verified'])
        self.assertTrue(entry.read(self.args.out/'completion.json')['registry_persisted'])
        self.assertEqual(list((self.args.out/'work/module').glob('*.runtime-plan.json')),[])

    def test_offline_environment_telemetry_private_paths_and_pinned_dependencies(self):
        with patch.dict(os.environ,{'HOME':'/existing-home','GOFLAGS':'foreign','GOPROXY':'foreign','LD_PRELOAD':'foreign','NIX_PATH':'foreign'}):
            self.args.run=False;result=self.run_fake()
        env=entry.environment(self.args.out)
        self.assertEqual(env['GOTOOLCHAIN'],'local');self.assertEqual(env['GOPROXY'],'off');self.assertEqual(env['GOSUMDB'],'off')
        self.assertEqual(env['GOMAXPROCS'],'1');self.assertEqual(env['GOMEMLIMIT'],'2GiB')
        self.assertNotIn('GOFLAGS',env);self.assertNotIn('LD_PRELOAD',env)
        self.assertEqual((self.args.out/'work/telemetry/mode').read_text().strip(),'off')
        mod=(self.args.out/'work/module/go.mod').read_text()
        self.assertIn('module pt-agent-qwen-native-components',mod);self.assertIn('go 1.25.0',mod)
        self.assertIn('require github.com/tuneinsight/lattigo/v6 v6.0.0',mod)
        self.assertIn('replace github.com/tuneinsight/lattigo/v6 => '+str(self.lattigo),mod)
        self.assertEqual(result['status'],'prepared');self.assertEqual(self.calls,[])
        self.assertFalse((self.args.out/'work/module-cache').exists())

    def test_build_error_preserves_actual_observation_and_never_retries(self):
        bad=dict(observation(),exit_code=1)
        result=self.run_fake(lambda *a,**k:bad)
        self.assertEqual(result['status'],'error');self.assertEqual(len(self.calls),1)
        self.assertEqual(result['observation'],bad);self.assertEqual(result['completed_builds'],0)
        self.assertFalse(result['generic_library_compile_accepted']);self.assertTrue((self.args.out/'completion.json').is_file())

    def test_fake_build_explicitly_disables_saved_go_environment_and_parent_workspace(self):
        with patch.dict(os.environ,{'GOENV':'/foreign/go-env','GOWORK':'/foreign/go.work'}):
            result=self.run_fake()
        self.assertEqual(result['status'],'ok');self.assertEqual(len(self.calls),1)
        env=self.calls[0][1]['env']
        self.assertEqual(env['GOENV'],'off');self.assertEqual(env['GOWORK'],'off')

    def test_clean_exit_without_archive_is_explicit_error(self):
        result=self.run_fake(lambda *a,**k:observation())
        self.assertEqual(result['status'],'error');self.assertEqual(result['completed_builds'],1)
        self.assertFalse(result['generic_library_build_passed']);self.assertIn('archive missing',result['error'])

    def test_nonclean_signal_or_residual_process_rejected(self):
        for change in ({'signals_sent':[15]},{'remaining_pids':[123]},{'signal':15},{'timed_out':True}):
            with self.subTest(change=change):
                with self.assertRaises(RuntimeError):entry.clean(dict(observation(),**change))

    def test_source_failure_closes_result_registry_and_starts_no_build(self):
        def gate(args):raise ValueError('original SDK absent')
        result=self.run_fake(gate=gate)
        self.assertEqual(result['status'],'error');self.assertEqual(self.calls,[])
        self.assertFalse(result['linked_native_program_verified']);self.assertTrue((self.args.out/'completion.json').is_file())

    def test_physical_4GiB_and_available2GiB_guards_are_independent(self):
        self.assertFalse(entry.physical(dict(subtree_rss_bytes=4*entry.GIB),2*entry.GIB)['stop'])
        self.assertTrue(entry.physical(dict(subtree_rss_bytes=4*entry.GIB+1),3*entry.GIB)['stop'])
        self.assertTrue(entry.physical(None,2*entry.GIB-1)['stop'])
        with patch.object(entry,'physical',return_value=dict(stop=True,reasons=['available reserve'])):
            result=self.run_fake()
        self.assertEqual(result['status'],'error');self.assertEqual(self.calls,[])

    def test_post_build_guard_preserves_real_completion_and_revokes_acceptance(self):
        def physical(sample,available):return dict(stop=len(self.calls)==1,reasons=['post-build guard'])
        with patch.object(entry,'physical',side_effect=physical):result=self.run_fake()
        self.assertTrue(result['generic_library_build_passed']);self.assertEqual(result['completed_builds'],1)
        self.assertEqual(result['status'],'error');self.assertFalse(result['generic_library_compile_accepted'])

    def test_terminal_case_limit_revokes_generic_acceptance_and_keeps_all_link_flags_false(self):
        self.args.out.mkdir();result=dict(status='ok',attempted_builds=1,completed_builds=1,generic_library_build_passed=True)
        with patch.object(entry,'artifact_bytes',return_value=entry.CASE_BYTES), \
                patch.object(entry,'persist_case_completion',return_value=dict(case_artifact_bytes=0)):
            entry.close(self.args.out,result,1,entry.StageRecorder(self.args.out/'events.jsonl'),self.args.registry)
        self.assertEqual(result['status'],'error');self.assertFalse(result['generic_library_compile_accepted'])
        self.assertFalse(result['linked_native_program_verified']);self.assertFalse(result['full_qwen_compile_verified'])

    def test_original52_call_and3_concrete_bootstrap_connector_source(self):
        run=function(self.go,'RunTwoTokens');forward=function(self.go,'forward');boot=function(self.go,'BootstrapBoundary');native=function(self.go,'nativeBoot')
        self.assertEqual(run.count('s.forward(p,'),2);self.assertIn('for _,ref:=range p.Layers',forward)
        self.assertIn('Layers [24]ComponentRef',self.go);self.assertEqual(forward.count('s.RunComponent('),2)
        self.assertEqual(run.count('s.RunComponent('),2) # first and terminal, one invocation each.
        self.assertEqual(2*(24+1)+2,52)
        self.assertEqual(run.count('s.BootstrapBoundary(chosen0['),3)
        self.assertIn('input.value.CopyNew()',boot);self.assertIn('copied.Level()-1',boot);self.assertIn('s.nativeBoot(copied)',boot)
        self.assertIn('s.bootstrapper.Bootstrap(input)',native);self.assertIn('DropLevel(out,out.Level()-5)',native)
        self.assertNotIn('Scale =',boot);self.assertNotIn('bootstrapInputDivisor',self.go)

    def test_original_token0_prefix_tail_masks_and_first19_lifetime_source(self):
        run=function(self.go,'RunTwoTokens');forward=function(self.go,'forward')
        self.assertIn('firstInputs[19]=original[0]',run)
        self.assertIn('second:=original',run)
        self.assertIn('second[p.PrefixLength],second[32],second[33]=mixed,attention,selector',run)
        self.assertIn('copy(result[:19],chosen0[:19])',run);self.assertIn('copy(result[19:],chosen1)',run)
        self.assertIn('next[32],next[33]=current[32],current[33]',forward)
        self.assertNotIn('second:=current',run);self.assertNotIn('second:=logits0',run)

    def test_independent_multi_IO_SSA_cache_all_final_and_actual_typed_arithmetic_source(self):
        loader=function(self.go,'LoadComponent');run=function(self.go,'RunComponent');operation=function(self.go,'validateOperation')
        self.assertIn('len(p.Inputs)!=ni',loader);self.assertIn('len(p.Outputs)!=no',loader)
        self.assertIn('seen:=map[string]bool{}',loader);self.assertIn('ordinals:=map[int]bool{}',loader)
        self.assertIn('pool:=map[string]*rlwe.Ciphertext{}',run);self.assertIn('cache:=plaintextCache{shared:s,component:c}',run)
        self.assertIn('for _,id:=range c.plan.Outputs{final[id]=true}',run)
        self.assertIn('!final[id]',run);self.assertIn('!final[op.Output]',run)
        self.assertNotIn('Outputs[0]',run);self.assertIn('owner!=s',function(self.go,'require'))
        for op in ('mod_switch','rotate','mul_cp','rescale','add_cc','add_cp','boot','mul_cc','relinearize','negate'):
            self.assertIn('case "'+op+'":',run)
        self.assertIn('op.Attrs.OperatorProfile!=BootstrapProfile',operation)
        self.assertIn('op.Attrs.Scale!=scale-43',operation)
        self.assertIn('inputUnchanged(a,aBefore)',run);self.assertIn('validateCipher(out,v,predicted)',run)

    def test_payload_unsupported_errors_and_no_parameter_key_or_decryption_constructor(self):
        self.assertIn('only finite singleton inline payload',self.go)
        self.assertIn('require original full-slot float64 bundle vector',self.go)
        self.assertIn('p.DefaultScale().Log2()!=43',function(self.go,'BorrowSharedContext'))
        self.assertNotRegex(self.go,r'\b(?:NewParametersFromLiteral|NewKeyGenerator|NewEncryptor|NewDecryptor|EncryptNew|DecryptNew)\s*\(')
        self.assertNotRegex(self.go,r'\bfunc\s+(?:main|init)\s*\(')
        self.assertNotIn('map_precision_plan',self.go)

    def test_cli_has_no_fake_component_manifest_or_runtime_entry(self):
        args=entry.parser().parse_args(['--out',str(self.args.out),'--run'])
        self.assertTrue(args.run);self.assertFalse(hasattr(args,'manifest'));self.assertFalse(hasattr(args,'execute'))
        self.assertEqual(entry.TOTAL_SECONDS,900);self.assertEqual(entry.BUILD_SECONDS,600)

    def test_original_encode_place_is_on_value_while_compute_place_remains_strict(self):
        loader=function(self.go,'LoadComponent');operation=function(self.go,'validateOperation')
        encode=loader[loader.index('for _,op:=range p.Initialization {'):loader.index('for n,op:=range p.Execution {')]
        self.assertNotIn('validPlace(op.Place)',encode)
        self.assertIn('!validValue(v)',loader);self.assertIn('!validPlace(op.Place)',operation)


if __name__=='__main__':unittest.main()
