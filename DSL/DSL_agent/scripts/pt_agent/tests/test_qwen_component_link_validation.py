"""Synthetic metadata and fake native-client observations only; no Go/tools."""
from argparse import Namespace
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_component_link_validation as entry


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n' if isinstance(value, dict) else value)


def state():
    return dict(step=0, prefix_length=9, prefix_token_ids=list(range(32)), sample_id='train:0',
        attention_mask=[[1.]*9+[0.]*23], sel=[[0.]*8+[1.]+[0.]*23])


def components(root):
    rows = {}
    for name in [*('layer'+str(i) for i in range(24)), 'head', 'first', 'terminal']:
        role = 'layer' if name.startswith('layer') else name
        index = int(name[5:]) if role == 'layer' else 0
        ni, no = (34, 32) if role == 'layer' else (34, 19) if role == 'head' else (20, 22) if role == 'first' else (19, 19)
        rows[name] = dict(role=role, index=index, plan=str(root/name/'original.runtime-plan.json'),
            bundle=str(root/name/'original.bundle'), format_version=1,
            target=dict(profile='qwen48', slots=16384),
            inputs=[dict(local_id=str(i), descriptor=deepcopy(entry.TYPE)) for i in range(ni)],
            outputs=[dict(local_id=str(ni+i), descriptor=deepcopy(entry.TYPE)) for i in range(no)],
            native_boots=no if role in ('layer', 'head') else 7,
            native_operation_count=2*no, native_op_counts=dict(boot=no if role in ('layer', 'head') else 7))
    for name in ('first', 'terminal'):
        for i, row in enumerate(rows[name]['outputs']):
            row['descriptor']['level'] = i % 5+1
        rows[name]['external_boundary_bootstraps_in_component'] = 0
    return rows


def consumed(manifest):
    program = manifest['program']; refs = [*program['Layers'], program['Head'], program['First'], program['Terminal']]
    return dict(status='ok', error=None, program_validations=1, component_load_attempts=27,
        component_loads_completed=27, program_and_original27_artifacts_consumed=True,
        whole_generation_executed=False,
        actual_library_source_AST=dict(component_calls_encoded=52, boundary_boots_encoded=3,
            slots=16384, context='qwen-bootstrap48', nominal_scale_log2=43, ring_degree=65536),
        components=[dict(role=r['Role'], index=r['Index'], plan=r['PlanPath'], bundle=r['BundlePath'],
            source_case=r['SourceCase']) for r in refs],
        **dict.fromkeys(('payload_reads','context_constructions','parameter_constructions','keygen_runs',
            'RunComponent_calls','RunTwoTokens_calls','encryptions','decryptions','bootstraps'), 0))


class TypedLinkChecks(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.parts = components(self.root)

    def link(self):
        return entry.link_manifest(self.parts, state(), self.root/'frozen')

    def test_complete52_actual_call_namespaces_and38_real_outputs(self):
        manifest = self.link(); calls = manifest['calls']
        self.assertEqual(len(calls), 52)
        self.assertEqual([c['component'] for c in calls[:26]], [*('layer'+str(i) for i in range(24)), 'head', 'first'])
        self.assertEqual([c['component'] for c in calls[26:]], [*('layer'+str(i) for i in range(24)), 'head', 'terminal'])
        self.assertEqual(len({c['local_SSA_namespace'] for c in calls}), 52)
        self.assertEqual(len({v['id'] for v in manifest['final_outputs']}), 38)
        self.assertEqual(manifest['explicit_boundary_requests'], 1577)
        self.assertEqual(manifest['native_boots_encoded'], 2*(24*32+19)+14+3)
        self.assertFalse(manifest['full_generation_executed'])
        self.assertFalse(manifest['actual_ciphertext_metadata_verified'])

    def test_original_token0_prefix_tail_masks_and_single_consumed_embedding(self):
        manifest = self.link(); calls = manifest['calls']
        self.assertEqual(calls[25]['inputs'][-1], 'original:0')
        expected = ['original:'+str(i) for i in range(34)]
        expected[9], expected[32], expected[33] = 'connector:mixed', 'connector:attention', 'connector:selector'
        self.assertEqual(manifest['second_forward_initial'], expected)
        self.assertEqual(calls[26]['inputs'], expected)
        for i in range(1, 24):
            self.assertEqual(calls[i]['inputs'][-2:], ['original:32', 'original:33'])
            self.assertEqual(calls[26+i]['inputs'][-2:], ['connector:attention', 'connector:selector'])
        self.assertEqual(manifest['final_outputs'][:19], [dict(id='call25:first:'+v['local_id'], descriptor=v['descriptor'])
            for v in self.parts['first']['outputs'][:19]])
        self.assertEqual(manifest['calls'][-1]['component'], 'terminal')

    def test_onehot_output_levels_preserved_and_only3_connector_boots(self):
        manifest = self.link()
        self.assertEqual([v['descriptor']['level'] for v in manifest['final_outputs'][:19]], [i % 5+1 for i in range(19)])
        self.assertEqual(len(manifest['connectors']), 3)
        self.assertEqual([c['input_descriptor']['level'] for c in manifest['connectors']], [5, 1, 2])
        self.assertEqual([c['drop_levels_before_native_boot'] for c in manifest['connectors']], [4, 0, 1])
        self.assertTrue(all(c['output_descriptor'] == entry.TYPE for c in manifest['connectors']))
        self.assertEqual(self.parts['first']['outputs'][19]['descriptor']['level'], 5)

    def test_missing_component_wrong_parameter_target_or_embedded3_are_rejected(self):
        saved = deepcopy(self.parts)
        del self.parts['layer23']
        with self.assertRaises(ValueError): self.link()
        self.parts = deepcopy(saved); self.parts['terminal']['target']['slots'] = 8192
        with self.assertRaises(ValueError): self.link()
        self.parts = deepcopy(saved); self.parts['first']['external_boundary_bootstraps_in_component'] = 3
        with self.assertRaisesRegex(ValueError, 'double counting'): self.link()

    def test_incompatible_actual_descriptor_and_mask_state_rejected(self):
        self.parts['head']['outputs'][0]['descriptor']['level'] = 4
        with self.assertRaisesRegex(ValueError, 'consumer'): self.link()
        self.parts = components(self.root)
        invalid = state(); invalid['attention_mask'][0][31] = 1.
        with self.assertRaises(ValueError): entry.link_manifest(self.parts, invalid, self.root/'frozen')
        self.parts['first']['outputs'][0]['descriptor']['scale_log2'] = 50
        with self.assertRaisesRegex(ValueError, 'raw43'): self.link()

    def test_distinct_canonical_artifacts_and_original_layer_roles_are_required(self):
        self.parts['layer1']['plan'] = str(self.root/'layer0/sub/../original.runtime-plan.json')
        with self.assertRaisesRegex(ValueError, 'canonical'): self.link()
        self.parts = components(self.root); self.parts['layer23']['index'] = 22
        with self.assertRaisesRegex(ValueError, 'order'): self.link()

    def test_own_supervisor_is_in_the_physical4GiB_guard(self):
        row = entry._physical(dict(subtree_rss_bytes=4*entry.GIB-100), 2*entry.GIB, own_rss=100)
        self.assertFalse(row['stop']);self.assertEqual(row['subtree_RSS_bytes'],4*entry.GIB)
        self.assertTrue(entry._physical(dict(subtree_rss_bytes=4*entry.GIB-99),2*entry.GIB,own_rss=100)['stop'])

    def test_metadata_consumption_requires_exact27_paths_and_zero_runtime_actions(self):
        manifest = self.link(); good = consumed(manifest)
        entry.check_consumed(good, manifest)
        for change in ('payload', 'source', 'calls', 'bool_counter'):
            bad = deepcopy(good)
            if change == 'payload': bad['payload_reads'] = 1
            elif change == 'source': bad['components'][0]['source_case'] = str(self.root/'foreign')
            elif change == 'calls': bad['actual_library_source_AST']['component_calls_encoded'] = 50
            else: bad['program_validations'] = True
            with self.subTest(change=change), self.assertRaises(ValueError): entry.check_consumed(bad, manifest)

    def test_streamed_metadata_retains_actual_levels_and_rejects_duplicate_outputs(self):
        p = self.root/'kernel.runtime-plan.json'; row = self.parts['first']
        data = dict(format_version=1, target=row['target'], external_inputs=[v['local_id'] for v in row['inputs']],
            final_outputs=[v['local_id'] for v in row['outputs']], initialization=[], finalization=[],
            values=[dict(id=v['local_id'], **v['descriptor']) for v in row['inputs']+row['outputs']],
            execution=[dict(op='boot') for _ in range(7)])
        put(p, data)
        actual = entry.component_metadata(p, 'first', 0, 7)
        self.assertEqual(actual['outputs'], row['outputs']); self.assertEqual(actual['native_boots'], 7)
        data['final_outputs'][-1] = data['final_outputs'][0]; put(p, data)
        with self.assertRaises(ValueError): entry.component_metadata(p, 'first', 0, 7)

    def test_client_is_metadata_only_and_ties_actual_caller_AST(self):
        self.assertIn('in.Program.Validate()', entry.CLIENT); self.assertIn('q.LoadComponent(ref)', entry.CLIENT)
        self.assertIn('runtime.GC()', entry.CLIENT); self.assertIn('parser.ParseFile', entry.CLIENT)
        self.assertIn('s.forward', entry.CLIENT); self.assertIn('firstInputs[19]=original[0]', entry.CLIENT)
        self.assertIn('second[p.PrefixLength],second[32],second[33]=mixed,attention,selector', entry.CLIENT)
        self.assertNotRegex(entry.CLIENT, r'\b(?:RunTwoTokens|RunComponent|BorrowSharedContext|NewParametersFromLiteral|NewKeyGenerator|EncryptNew|DecryptNew)\s*\(')
        self.assertNotIn('readVector(', entry.CLIENT)


class AdmissionOuterChecks(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = Namespace(out=self.root/'fresh-link', registry=self.root/'registry.jsonl', run=True,
            source_case=self.root/'frozen', probe_case=self.root/'probe', remaining_case=self.root/'remaining',
            kernel_case=self.root/'kernels', library_case=self.root/'library', initialization_audit=self.root/'init.json')
        self.calls = []; self.library_fixture(); self.manifest = entry.link_manifest(components(self.root), state(), self.args.source_case)
        self.checked = dict(library=entry.library_gate(self.args.library_case), manifest=self.manifest)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(entry.paths, 'RESULTS_DIR', self.root))
        self.stack.enter_context(patch.object(entry, '_available_ram_bytes', return_value=12*entry.GIB))
        self.stack.enter_context(patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=30*entry.GIB)))

    def library_fixture(self):
        case = self.args.library_case; source = dict(go_version='1.25.13', lattigo_version='6.0.0',
            sdk=str(self.root/'sdk'), lattigo_source=str(self.root/'lattigo'), parameters_selected=False,
            actual_component_artifacts_admitted=0)
        put(case/'work/module/go.mod', 'module pt-agent-qwen-native-components\n\ngo 1.25.0\n')
        put(case/'work/module/go.sum', 'public existing module metadata\n')
        put(case/'work/module/runtime.go', 'package qwengeneration\n// synthetic metadata fixture only\n')
        put(case/'work/qwen-generation.a', 'do not read this fake archive payload\n')
        for name in ('module-cache','build-cache'): put(case/'work'/name/'fixture', 'tiny public fixture\n')
        self.result = dict(round='QWEN-COMPONENT-LIBRARY-CPU-OVERLAP', mode='offline-generic-library-only-during-identified-bert-tune3',
            status='ok', generic_library_build_passed=True, generic_library_compile_accepted=True,
            attempted_builds=1, completed_builds=1, linked_native_program_verified=False, full_qwen_compile_verified=False,
            whole_generation_verified=False, source=source, archive=dict(path=str(case/'work/qwen-generation.a'), bytes=999999999),
            observation=dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[]), **dict.fromkeys(entry.ZERO,0))
        put(case/'result.json', self.result)
        put(case/'completion.json', dict(status='ok', registry_persisted=True, required_artifacts_persisted=True))
        reviewed = self.root/'reviewed.go'; put(reviewed, 'package reviewed\n')
        put(self.args.initialization_audit, dict(format='qwen-component-import-initialization-source-audit-v1',
            status='passed_source_no_model_context_keys_or_payload_initializers', roots=dict(lattigo=source['lattigo_source'],sdk=source['sdk']),
            lattigo_version='6.0.0', unresolved_imports=[], audited_files=[dict(path=str(reviewed))],
            actual_library_initializer=dict(path=str(case/'work/module/runtime.go')),actual_library_imports=[],
            explicit_review=dict.fromkeys(('model_context_initializers','parameter_constructor_initializers',
                'keygen_initializers','external_payload_reads'),0)))

    def fake(self, fail=None, bad=None):
        def process(command, **options):
            phase = Path(options['log']).stem; self.calls.append((phase,command,options))
            Path(options['log']).write_text('fake metadata helper only\n'); Path(options['resources']).write_text('')
            if phase == 'build': put(self.args.out/'qwen-component-inspector', 'synthetic binary placeholder\n')
            else:
                native = consumed(self.manifest)
                if bad: native.update(bad)
                put(self.args.out/'native-consumption.json', native)
            return dict(exit_code=1 if fail == phase else 0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])
        with patch.object(entry, 'source_gate', return_value=deepcopy(self.checked)), patch.object(entry, 'run_process', side_effect=process):
            return entry.execute(self.args)

    def test_actual_library_gate_uses_paths_and_status_without_archive_read_or_size_comparison(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('No archive integrity read')):
            accepted = entry.library_gate(self.args.library_case)
        self.assertEqual(accepted['archive']['bytes'], 999999999)
        self.result['attempted_builds'] = True;put(self.args.library_case/'result.json',self.result)
        with self.assertRaises(ValueError): entry.library_gate(self.args.library_case)
        self.result['attempted_builds'] = 1
        self.result['archive']['path'] = str(self.root/'foreign.a'); put(self.args.library_case/'result.json', self.result)
        with self.assertRaises(ValueError): entry.library_gate(self.args.library_case)

    def test_import_initializer_review_is_explicit_same_version_and_source_paths(self):
        accepted = entry.initialization_gate(self.args.initialization_audit, self.checked['library'])
        self.assertEqual(accepted['source_files'], 1)
        audit = entry.read(self.args.initialization_audit); audit['explicit_review']['parameter_constructor_initializers'] = 1
        put(self.args.initialization_audit, audit)
        with self.assertRaises(ValueError): entry.initialization_gate(self.args.initialization_audit, self.checked['library'])

    def test_full_fake_build_and_consume_have_finite_limits_and_unique_durable_terminal(self):
        result = self.fake()
        self.assertEqual(result['status'], 'ok'); self.assertTrue(result['constructor_free_program_and_artifacts_accepted'])
        self.assertEqual(result['completed'], dict(build=1, consume=1))
        self.assertEqual([row[0] for row in self.calls], ['build','consume'])
        for (phase,argv,opts),limit in zip(self.calls, (600,900)):
            self.assertEqual(opts['mem_gib'],4); self.assertEqual(opts['fsize_gib'],8); self.assertLessEqual(opts['timeout'],limit)
            self.assertEqual(opts['env']['GOPROXY'],'off'); self.assertEqual(opts['env']['GOENV'],'off')
        self.assertIn('-mod=readonly',self.calls[0][1]); self.assertIn('--manifest',self.calls[1][1])
        self.assertEqual((self.args.out/'work/module/componentlib/runtime.go').read_text(), 'package qwengeneration\n// synthetic metadata fixture only\n')
        self.assertFalse(result['linked_native_program_executed']); self.assertFalse(result['full_generation_executed'])
        self.assertTrue(all(result[k] == 0 for k in entry.ZERO)); self.assertIsNone(result['CKKS_latency'])
        row = entry.read(self.args.registry)
        self.assertEqual(set(row),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        marker = entry.read(self.args.out/'completion.json')
        self.assertTrue(marker['registry_persisted']); self.assertTrue(marker['required_artifacts_persisted'])

    def test_first_build_failure_keeps_real_observation_and_skips_consume(self):
        result = self.fake(fail='build')
        self.assertEqual(result['status'],'error'); self.assertEqual(len(self.calls),1)
        self.assertEqual(result['attempted'],dict(build=1,consume=0)); self.assertEqual(result['completed'],dict(build=0,consume=0))
        self.assertFalse(result['constructor_free_program_and_artifacts_accepted'])
        self.assertEqual(result['processes'][0]['observation']['exit_code'],1)

    def test_native_bad_metadata_and_missing_initializer_audit_reject_without_acceptance(self):
        result = self.fake(bad=dict(RunTwoTokens_calls=1))
        self.assertEqual(result['status'],'error'); self.assertEqual(result['completed']['consume'],0)
        self.assertFalse(result['constructor_free_program_and_artifacts_accepted'])
        self.args.out = self.root/'second-fresh-link'; self.calls.clear(); self.args.initialization_audit.unlink()
        result = self.fake()
        self.assertEqual(result['status'],'error'); self.assertEqual(self.calls,[])

    def test_prepare_only_does_not_build_or_require_runtime_initializer_admission(self):
        self.args.run = False; self.args.initialization_audit.unlink()
        result = self.fake()
        self.assertEqual(result['status'],'prepared'); self.assertEqual(self.calls,[])
        self.assertFalse(result['constructor_free_program_and_artifacts_accepted'])
        self.assertTrue((self.args.out/'global-link-manifest.json').is_file())
        self.assertFalse((self.args.out/'work/module-cache').exists())

    def test_terminal_budget_revokes_acceptance_and_retains_completed_counts(self):
        self.args.out.mkdir(); result = dict(status='ok',attempted=dict(build=1,consume=1),completed=dict(build=1,consume=1))
        with patch.object(entry,'artifact_bytes',return_value=entry.CASE_BYTES), \
                patch.object(entry,'persist_case_completion',return_value=dict(case_artifact_bytes=0)):
            entry._close(self.args,result,entry.time.perf_counter_ns())
        self.assertEqual(result['status'],'error'); self.assertFalse(result['constructor_free_program_and_artifacts_accepted'])
        self.assertEqual(result['completed']['consume'],1)


    def test_source_gate_reuses_all_actual_component_admissions_and_same_C2_binding(self):
        from pt_agent.experiments import qwen_remaining_prefill_compile as remaining
        from pt_agent.experiments import qwen_generation_kernel_compile as kernel
        rows = components(self.root); frozen = dict(source_case=str(self.args.source_case), profile='original qwen48',
            original_source=dict(recovery=dict(pinned_error_counts={8:43})))
        checked = dict(frozen_source=frozen, first_layer_runtime_plan=rows['layer0']['plan'],
            first_layer_expected_bootstraps=32)
        prefill = {}
        for name in remaining.COMPONENTS:
            row = rows[name]
            prefill[name] = dict(status='ok',case=str(self.root/name),runtime_plan=row['plan'],
                component=dict(role=row['role'], index=row['index']), expected_compiled_bootstraps=row['native_boots'])
        master = dict(round=remaining.ROUND,mode=remaining.MODE,status='ok',source=checked,
            full_prefill_verified=True,accepted_components=25,components=dict(prefill,layer0=dict(runtime_plan=rows['layer0']['plan'])))
        kchecked = dict(frozen_source=entry._json(frozen),initial_state=state(),binding=dict(real_source='synthetic fixture'),lineage={})
        krows = {name:dict(status='ok',case=str(self.root/name),runtime_plan=rows[name]['plan'],
            native=dict(compiled_bootstraps=7)) for name in ('first','terminal')}
        kmaster = dict(round=kernel.ROUND,mode=kernel.MODE,status='ok',source=kchecked,accepted_components=2,
            two_kernel_native_components_verified=True,components=krows)
        complete = dict(status='ok',registry_persisted=True,required_artifacts_persisted=True)
        put(self.args.remaining_case/'result.json',master);put(self.args.remaining_case/'completion.json',complete)
        put(self.args.kernel_case/'result.json',kmaster);put(self.args.kernel_case/'completion.json',complete)
        by_path = {r['plan']:r for r in rows.values()}
        with patch.object(remaining,'admission',return_value=checked) as first, \
                patch.object(remaining,'accept_component',side_effect=lambda case,name,proof:prefill[name]) as accepted, \
                patch.object(kernel,'admission',return_value=kchecked), \
                patch.object(kernel,'accept_component',side_effect=lambda case,name,proof:krows[name]) as kernels, \
                patch.object(entry,'component_metadata',side_effect=lambda path,*args:deepcopy(by_path[str(path)])):
            result = entry.source_gate(self.args)
            first.assert_called_once_with(self.args.source_case,self.args.probe_case)
            self.assertEqual(accepted.call_count,24);self.assertEqual(kernels.call_count,2)
            self.assertEqual(len(result['manifest']['calls']),52);self.assertEqual(result['payload_reads'],0)
            kchecked['frozen_source'] = dict(frozen,selected_iteration=999)
            put(self.args.kernel_case/'result.json',dict(kmaster,source=kchecked))
            with self.assertRaisesRegex(ValueError,'same accepted original frozen'):
                entry.source_gate(self.args)


if __name__ == '__main__':
    unittest.main()
