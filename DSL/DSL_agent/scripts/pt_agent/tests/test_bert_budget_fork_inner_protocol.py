"""Explicit outer proof in real trusted worker skeletons; numeric hooks unstarted."""
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent.experiments import bert_budget_fork_compile as entry
from pt_agent.tests import test_bert_budget_fork_compile as fixtures


class ForbiddenRoot:
    def resolve(self):
        raise AssertionError('Inner access to configured global results root is forbidden')


class InnerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ForkCompileTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.args=self.fixture.args;self.args.out.mkdir()
        self.outer=entry.namespace(self.args)
        self.source=self.outer['metadata_source'](self.args.source_case,self.args.checkpoint_audit)
        self.proof=self.args.out/'outer-admission.json'
        self.assertTrue(self.proof.is_file())

    def inner(self,phase):
        args=copy.copy(self.args);args.worker_phase=phase;args.delegate_compile=False
        args.admission_proof=self.proof;args.out=None;args.compile_out=None
        args.work_dir=self.fixture.root/('inner-'+phase)
        args.model_info=self.args.source_case/'model/model_info.json';args.program=self.args.source_case/'session/program.ptdsl'
        args.math_result=self.args.out/'accepted-math.json'
        return args

    def test_real_math_and_all_native_worker_source_paths_do_not_access_global_registry_or_root(self):
        proof=self.fixture.proof(self.outer)
        fixtures.fixture_module.put(self.args.out/'accepted-math.json',proof)
        original_open=Path.open;forbidden=[];hooks=[]
        def local_only(path,*args,**kwargs):
            if Path(path).name=='registry.jsonl':
                forbidden.append(str(path));raise AssertionError('Inner global registry access is forbidden')
            return original_open(path,*args,**kwargs)
        def numeric_unstarted(*args,**kwargs):
            hooks.append('math');raise RuntimeError('Numerical hook intentionally unstarted')
        with mock.patch.object(entry.paths,'RESULTS_DIR',ForbiddenRoot()),mock.patch.object(Path,'open',local_only), \
             mock.patch.object(entry.base.original,'native_metadata_commands',side_effect=lambda *a:nullcontext()):
            args=self.inner('math');scope=entry.namespace(args);scope['_math_api']=numeric_unstarted
            result=scope['math_worker'](args)
            self.assertEqual(result['source'],self.source)
            self.assertEqual(result['status'],'error');self.assertIn('intentionally unstarted',result['error'])
            self.assertEqual(result['attempted'],dict.fromkeys(entry.base.COUNTERS,0))
            for phase in ('emit','trace','compile'):
                args=self.inner(phase);scope=entry.namespace(args);compiled=scope['compile_namespace']()
                def unstarted(*a,**k):
                    hooks.append(phase);raise RuntimeError('Compiler hook intentionally unstarted')
                compiled['_emit']=unstarted;compiled['_child']=unstarted
                with mock.patch('pt_agent.toolchain.host_paths',return_value=dict(hecate='/synthetic',hecate_build='/synthetic',hecate_opt='/synthetic')), \
                     mock.patch.dict(os.environ,{'HECATE_PYTHON_LIBRARY_PATH':'/synthetic'}):
                    observed=compiled['worker'](args)
                self.assertEqual(observed['source']['fork_provenance'],self.source['fork_provenance'])
                self.assertEqual(observed['source']['mathematical_groups_passed'],3)
                self.assertEqual(observed['status'],'error');self.assertIn('intentionally unstarted',observed['error'])
        self.assertEqual(forbidden,[]);self.assertEqual(hooks,['math','emit','trace','compile'])

    def test_outer_duplicate_registry_still_rejects_before_any_child(self):
        path=self.fixture.root/'registry.jsonl';rows=path.read_text();path.write_text(rows+rows)
        with self.assertRaisesRegex(ValueError,'exactly one'):
            entry.fork_gate(self.args.source_case,self.args.checkpoint_audit,self.args)

    def test_inner_missing_or_foreign_proof_and_claim_fail_without_global_lookup(self):
        original=entry.base.read(self.proof);args=self.inner('math')
        with mock.patch.object(entry.paths,'RESULTS_DIR',ForbiddenRoot()):
            args.admission_proof=None
            with self.assertRaisesRegex(ValueError,'explicit'):entry.fork_gate(args.source_case,args.checkpoint_audit,args)
            args.admission_proof=self.proof
            for field in ('original_source_case','boundary_snapshot','claim_path','registry_row','registry_run_id'):
                bad=copy.deepcopy(original)
                if field=='registry_row':bad[field]['metrics']['new_api_calls']=1
                elif field=='registry_run_id':bad['registry_row']['run_id']='foreign-fork'
                else:bad[field]='/foreign'
                fixtures.fixture_module.put(self.proof,bad)
                with self.subTest(field=field),self.assertRaises(ValueError):
                    entry.fork_gate(args.source_case,args.checkpoint_audit,args)
            fixtures.fixture_module.put(self.proof,original)

    def test_every_child_command_gets_one_explicit_same_proof_path(self):
        for phase in ('math','compile'):
            argv=self.outer['_argv'](self.args,phase,self.args.out/'math-result.json',10000)
            self.assertEqual(argv.count('--admission-proof'),1)
            self.assertEqual(argv[argv.index('--admission-proof')+1],str(self.proof))
        args=self.inner('emit')
        for phase in ('emit','trace','compile'):
            argv=entry.namespace(args)['compile_namespace']()['phase_command'](args,phase,self.fixture.root/'work')
            self.assertEqual(argv.count('--admission-proof'),1)
            self.assertEqual(argv[argv.index('--admission-proof')+1],str(self.proof))

    def test_same_metadata_source_in_outer_and_inner_and_no_silent_proof_overwrite(self):
        args=self.inner('math')
        with mock.patch.object(entry.paths,'RESULTS_DIR',ForbiddenRoot()):
            self.assertEqual(entry.metadata_source(args.source_case,args.checkpoint_audit,args),self.source)
        proof=entry.base.read(self.proof);proof['foreign']=True;fixtures.fixture_module.put(self.proof,proof)
        with self.assertRaisesRegex(ValueError,'prior outer-admission'):
            self.outer['metadata_source'](self.args.source_case,self.args.checkpoint_audit)


if __name__=='__main__':unittest.main()
