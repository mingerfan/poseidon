"""New byte/number controls only: synthetic complex cells, no native execution."""
import copy
import itertools
import math
import os
from pathlib import Path
import struct
import unittest

from delivery_component import bound, canonical, sha
from first_output_assessment import CKKS
from test_derived_compiler_program_adapter import runtime_fixture, fragment
import complete_program_numerical_observer as observer
import derived_compiler_program_adapter as adapter


def fixture():
    original, data = runtime_fixture(); target, runtime = fragment(data)
    references = [1., -2., 3., -4., 1., -2.]
    physical = [dict(name=r['decoded_file'], slots=16384, bytes=262144, ciphertext_sha256=sha(r['decoded_file'].encode()))
        for r in data['selection']['outputs']]
    byte_receipt = bound(dict(format='poseidon-B-program-complex-decoder-byte-receipt-v2', decoder_source_sha256=adapter.DECODER_SHA,
        decoder_contract_binding='a'*64, identity=target['physical_identity'], outputs=target['decoder_outputs'],
        input_slot_period=8, compact_periodic_copies_are_complete=True, raw_nonfinite_bits_preserved=True,
        logical_values=target['logical_values'], explicitly_recorded_alias_values=target['periodic_alias_values'], physical_files=physical))
    _, decoder = adapter.pinned_B_modules()
    def cells(change=None):
        values={'final':[1.,-2.],'branch':[3.,-4.],'same_final':[1.,-2.]}; files={p['name']:p['ciphertext_sha256']for p in physical}
        for selected in decoder.selector_cells(target['decoder_outputs']):
            real, imaginary=values[selected['output']][selected['logical_index']],0.
            if change is not None: real,imaginary=change(selected,real,imaginary)
            raw=struct.pack('<dd',real,imaginary)
            yield dict(selected,real=real,imaginary=imaginary,raw_cell_hex_le=raw.hex(),
                ciphertext_sha256=files[selected['file']],both_components_finite=math.isfinite(real)and math.isfinite(imaginary))
    def check(stream=None, reference=None, target_value=None, receipt=None):
        t=target if target_value is None else target_value
        return observer.compare_decoded(t,t['binding'],byte_receipt if receipt is None else receipt,'a'*64,
            cells()if stream is None else stream,references if reference is None else reference)
    return original,data,target,byte_receipt,references,cells,check


def full_byte_fixture(directory):
    """Synthetic public documents/opaque bytes; no key, compiler or HE runs."""
    directory.mkdir(mode=0o700);decoded=directory/'decoded';ciphertext=directory/'ciphertext'
    decoded.mkdir();ciphertext.mkdir()
    original,data=runtime_fixture();selection=copy.deepcopy(data['selection']);selection.pop('binding')
    program=data['program'];identity=dict(program_binding=program['binding'],case_binding=selection['case_binding'],
        input_bank_binding=selection['input_bank_binding'],key=program['key'])
    def artifact(name,raw):
        p=directory/name
        with p.open('xb')as f:f.write(raw)
        return dict(path=str(p),sha256=sha(raw))
    def document(name,value):return artifact(name,canonical(value))
    selected_units={r['unit_id']for r in selection['outputs']};manifests={};terminals=[];files=[]
    for unit in program['units']:
        records=[]
        for index,selected in enumerate(selection['outputs']):
            if selected['unit_id']!=unit['unit_id']:continue
            name='inert-final-%03d.ct'%index;payload=b'INERT PUBLIC CIPHERTEXT FIXTURE '+str(index).encode()
            root=ciphertext/unit['unit_id'];root.mkdir(exist_ok=True);(root/name).write_bytes(payload)
            record=dict(name=name,bytes=len(payload),sha256=sha(payload),level=3,scale=float(2**45))
            records.append(dict(value_id=selected['local_value_id'],record=record))
            selected.update(ciphertext_file=unit['unit_id']+'/'+name,decoded_file='final-%03d.actual-complex.f64'%index)
            pattern=([1.,-2.]if index==0 else[3.,-4.])+[0.]*6
            raw=b''.join(struct.pack('<dd',pattern[i%8],0.)for i in range(16384));(decoded/selected['decoded_file']).write_bytes(raw)
            files.append(dict(name=selected['decoded_file'],**{k:selected[k]for k in ['unit_id','local_value_id','unit_plan_sha256','logical']},
                slots=16384,bytes=len(raw),sha256=sha(raw),ciphertext_sha256=sha(payload),actual_level=3,actual_scale=float(2**45),nonfinite_values=0))
        desc=None
        if unit['unit_id']in selected_units:
            manifest=dict(format='poseidon-valid-epoch-native-cycle-output-v1',degree=32768,q=[45,51],p=[51],key_steps=[],
                plan_sha256=unit['plan_sha256'],epoch=program['key']['epoch'],origin=program['key']['origin'],
                candidate_library_sha256=program['key']['library_sha256'],files=records)
            desc=document(unit['unit_id']+'-manifest.json',manifest);manifests[unit['unit_id']]=dict(unit_id=unit['unit_id'],**desc)
        terminals.append(bound(dict(format='poseidon-program-unit-terminal-v1',status='succeeded',unit_id=unit['unit_id'],
            unit_plan_sha256=unit['plan_sha256'],protocol_sha256='b'*64,program_binding=program['binding'],
            case_binding=selection['case_binding'],input_bank_binding=selection['input_bank_binding'],
            output_manifest_sha256='c'*64 if desc is None else desc['sha256'],outputs=unit['outputs'])))
    for selected in selection['outputs']:selected['unit_output_manifest_sha256']=manifests[selected['unit_id']]['sha256']
    selection.update(terminals=terminals,private_client_executable_qualified=False,execution_authority=False)
    selection=bound(selection);data['selection']=selection;data['expected']['selection_binding']=selection['binding']
    target,_=fragment(data);selection_desc=document('selection.json',selection)
    protocol=dict(format='poseidon-program-final-client-protocol-v1',program_binding=program['binding'],case_binding=selection['case_binding'],
        **{k:program['key'][k]for k in ['epoch','origin','public_manifest_sha256']},candidate_library_sha256=program['key']['library_sha256'],
        units=[dict(unit_id=u['unit_id'],plan_sha256=u['plan_sha256'],protocol_sha256='b'*64)for u in program['units']])
    protocol_desc=document('protocol.json',protocol)
    files.sort(key=lambda row:row['name'])
    report=dict(status='program_final_outputs_decoded_numeric_metrics_pending',final_selection_sha256=selection_desc['sha256'],
        **{k:identity[k]for k in ['program_binding','case_binding','input_bank_binding']},
        **{k:identity['key'][k]for k in ['epoch','origin']},intermediate_decryptions=0,encryptions=0,new_keys=0,private_loader_calls=1,
        no_reference_in_client=True,all_selected_ciphertexts_loaded_before_private_loader=True,private_hash_exported=False,
        private_manifest_hash_computed_inside_client_only=True,protocol_sha256=protocol_desc['sha256'],
        client_final_decryptions=len(files),files=files)
    decoder_contract=bound(dict(format='poseidon-program-complex-final-output-decoder-v2',identity=identity,
        selection=selection_desc,client_report=document('client-report.json',report),client_protocol=protocol_desc,
        decoded_root=str(decoded),ciphertext_root=str(ciphertext),output_manifests=list(manifests.values()),
        outputs=target['decoder_outputs'],input_slot_period=8))
    source=artifact('candidate.py',original['source']);front=artifact('frontend.py',original['source']);native=artifact('native.py',original['source'])
    io=document('io-map.json',target['logical_IO']);dag=document('dag.json',target['compiler_program_map'])
    contract=bound(dict(format='poseidon-program-complex-assessment-target-v3',
        request=dict(host_binding='d'*64,public_binding='e'*64,case_binding=identity['case_binding']),
        model=document('model.json',original['request']['model']),
        inputs=[dict(name=r['name'],raw=dict(shape=r['shape'],dtype='|u1',bytes=2,sha256='f'*64),
            declared=dict(shape=r['shape'],dtype='<f8',bytes=16,sha256='0'*64),source_input_evidence_kind='retained_source_reconstruction')
            for r in original['request']['model']['inputs']],outputs=target['target_outputs'],
        first_source=dict(source=source,generation_receipt_binding='1'*64,trace_binding='2'*64,origin='retained_development'),
        compile=dict(dsl_frontend_input=front,native_input=native,native_input_kind='python_dsl',lowering_audit_binding='3'*64,
            record=artifact('compile.json',b'{"INERT_ONLY":true}'),io_map=io,io_map_binding=target['logical_IO']['binding'],
            compiler_dag=dag,compiler_dag_binding=target['compiler_program_map']['binding'],input_slot_period=8),
        decoder=document('decoder.json',decoder_contract),decoder_binding=decoder_contract['binding'],physical_identity=identity))
    return contract,target


class CompleteComplexControls(unittest.TestCase):
    def test_01_complete_real_imaginary_and_all_named_aliases(self):
        *_,check=fixture();r=check();c=r['comparison']
        self.assertEqual((c['compared_values'],c['compared_real_and_imaginary_components']),(12288,24576))
        self.assertEqual((r['logical_values'],r['logical_alias_values'],r['periodic_alias_values']),(6,2,12282))
        self.assertTrue(r['CKKS_observations_within_tolerance']);self.assertEqual(c['RMSE'],0.)
        self.assertFalse(any(r[k]for k in ['raw_numeric_passed','raw_joint_passed','first_numeric_passed','first_joint_passed']))
        root=Path(os.environ['POSEIDON_SYNTHETIC_CONTROL_ROOT'])/'complete-byte-fixture'
        contract,target=full_byte_fixture(root);bridge,_=adapter.pinned_B_modules()
        bridge_receipt,byte_receipt,stream=bridge.decode(contract,contract['binding'],adapter.DECODER,adapter.DECODER_SHA)
        numeric=observer.compare_decoded(target,target['binding'],byte_receipt,contract['decoder_binding'],stream(True),[1.,-2.,3.,-4.,1.,-2.])
        self.assertEqual(numeric['comparison']['compared_values'],12288);self.assertTrue(numeric['CKKS_observations_within_tolerance'])
        self.assertFalse(bridge_receipt['first_output_eligibility_or_numerical_success_granted'])

    def test_02_last_periodic_copy_imaginary_error_is_observed(self):
        *_,cells,check=fixture()
        def change(s,re,im):return(re,.01)if s['output']=='same_final'and s['alias_ordinal']==2047 and s['logical_index']==1 else(re,im)
        r=check(cells(change));self.assertEqual(r['comparison']['failed_values'],1)
        self.assertEqual(r['comparison']['compared_values'],12288);self.assertEqual(r['comparison']['maximum_imaginary_absolute_error'],.01)
        self.assertFalse(r['CKKS_observations_within_tolerance'])

    def test_03_joint_complex_norm_keeps_original_tolerance(self):
        *_,cells,check=fixture();tol=CKKS['atol']+CKKS['rtol']
        def change(s,re,im):return(re+.8*tol,.8*tol)if s['output']=='final'and s['alias_ordinal']==0 and s['logical_index']==0 else(re,im)
        r=check(cells(change));self.assertEqual(r['comparison']['failed_values'],1)
        self.assertGreater(r['comparison']['max_absolute_error'],tol)
        self.assertLess(r['comparison']['maximum_real_absolute_error'],tol)
        self.assertLess(r['comparison']['maximum_imaginary_absolute_error'],tol)

    def test_04_nonfinite_bits_count_as_failure_and_do_not_stop_stream(self):
        *_,cells,check=fixture()
        for bad in [float('inf'),float('-inf'),float('nan')]:
            def change(s,re,im):return(re,bad)if s['output']=='final'and s['alias_ordinal']==0 and s['logical_index']==0 else(re,im)
            r=check(cells(change));self.assertEqual(r['comparison']['nonfinite_native_values'],1)
            self.assertEqual(r['comparison']['compared_values'],12288);self.assertIsNone(r['comparison']['max_absolute_error'])
            self.assertIsNone(r['comparison']['RMSE']);self.assertFalse(r['CKKS_observations_within_tolerance']);canonical(r)
        with self.assertRaises(ValueError):check(reference=[float('inf'),-2.,3.,-4.,float('inf'),-2.])

    def test_05_model_value_alias_cannot_replace_reference(self):
        *_,check=fixture()
        with self.assertRaises(ValueError):check(reference=[1.,-2.,3.,-4.,1.,-2.001])
        with self.assertRaises(ValueError):check(reference=[1.,-2.,3.,-4.])

    def test_06_missing_extra_or_reordered_cells_reject(self):
        *_,cells,check=fixture()
        with self.assertRaises(ValueError):check(itertools.islice(cells(),12287))
        with self.assertRaises(ValueError):check(itertools.chain(cells(),itertools.islice(cells(),1)))
        rows=list(cells());rows[0],rows[1]=rows[1],rows[0]
        with self.assertRaises(ValueError):check(iter(rows))

    def test_07_raw_bytes_offsets_and_recorded_finiteness_are_bound(self):
        *_,cells,check=fixture()
        for field,value in [('raw_cell_hex_le',struct.pack('<dd',5.,0.).hex()),('imaginary_byte_offset',9),
                ('both_components_finite',False),('ciphertext_sha256','f'*64),('alias_ordinal',True)]:
            stream=cells();first=next(stream);first[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):check(itertools.chain([first],stream))

    def test_08_decoder_case_key_period_and_source_ids_cannot_drift(self):
        _,_,target,receipt,_,_,check=fixture()
        for mutate in [lambda d:d.update(decoder_source_sha256='f'*64),lambda d:d.update(decoder_contract_binding='f'*64),
                lambda d:d['identity'].update(case_binding='f'*64),lambda d:d['identity']['key'].update(epoch='f'*32),
                lambda d:d.update(input_slot_period=16),lambda d:d.update(explicitly_recorded_alias_values=12281)]:
            bad=copy.deepcopy(receipt);bad.pop('binding');mutate(bad);bad=bound(bad)
            with self.assertRaises(ValueError):check(receipt=bad)
        with self.assertRaises(ValueError):observer.compare_decoded(target,'f'*64,receipt,'a'*64,iter([]),[1.,-2.,3.,-4.,1.,-2.])

    def test_09_finite_overflow_is_failed_error_not_json_infinity(self):
        *_,cells,check=fixture()
        references=[-1e308,-2.,3.,-4.,-1e308,-2.]
        def change(s,re,im):return(1e308,0.)if s['output']in('final','same_final')and s['logical_index']==0 else(re,im)
        r=check(cells(change),reference=references);self.assertEqual(r['comparison']['nonfinite_errors'],4096)
        self.assertEqual(r['comparison']['compared_values'],12288);self.assertIsNone(r['comparison']['RMSE']);canonical(r)

    def test_10_every_policy_input_required_before_any_numeric_read(self):
        spec=bound(dict(input_bindings=[],test_only=True));policy=bound(dict(retained_mathematical_producer=spec,
            independent_math=dict(atol=1e-12,rtol=1e-12),CKKS=dict(CKKS),inputs=['1'*64,'2'*64]))
        for cases in [[],[dict(input_sha256='1'*64)], [dict(input_sha256='1'*64),dict(input_sha256='1'*64)],
                [dict(input_sha256='2'*64),dict(input_sha256='1'*64)]]:
            with self.assertRaises(ValueError):observer.observe_retained({},b'',policy,{},cases)
        bad=dict(policy);bad.pop('binding');bad['CKKS']=dict(atol=1e-3,rtol=1e-4)
        with self.assertRaises(ValueError):observer.observe_retained({},b'',bound(bad),{},[])

    def test_11_external_case_scope_binds_original_frozen_input_and_source(self):
        original,_,target,_,_,_,_=fixture()
        binding=dict(physical_input_sha256='3'*64,raw_input_sha256='4'*64,declared_input_sha256='5'*64,case_id='synthetic')
        contract=bound(dict(request=dict(host_binding='6'*64,public_binding='7'*64,case_binding='8'*64),
            physical_identity=target['physical_identity'],inputs=[],outputs=target['target_outputs'],
            first_source=dict(source=dict(path='/synthetic-only.py',sha256=sha(original['source'])),generation_receipt_binding='9'*64,trace_binding='a'*64),
            compile=dict(io_map_binding=target['logical_IO']['binding'],compiler_dag_binding=target['compiler_program_map']['binding'],input_slot_period=8)))
        case=dict(input_sha256='3'*64,target=contract,expected_target_binding=contract['binding'],derived_target=target,
            expected_derived_target_binding=target['binding'],expected_request=contract['request'],expected_identity=target['physical_identity'],
            expected_generation_receipt_binding='9'*64,expected_trace_binding='a'*64,expected_target_inputs=[],original_input_binding=binding)
        self.assertEqual(observer._case_scope(case,original['request'],original['source'],binding),(contract,target))
        for key,value in [('expected_trace_binding','f'*64),('expected_generation_receipt_binding','f'*64),
                ('expected_request',{}),('expected_identity',{}),('expected_target_inputs',[{}]),('original_input_binding',{})]:
            bad=copy.deepcopy(case);bad[key]=value
            with self.subTest(field=key),self.assertRaises(ValueError):observer._case_scope(bad,original['request'],original['source'],binding)

    def test_12_approximation_math_domain_application_and_whole_gates_are_separate(self):
        _,_,_,_,reference,_,check=fixture();native=check()
        policy=dict(independent_math=dict(atol=1e-12,rtol=1e-12),CKKS=dict(CKKS),frozen_domain_sha256='d'*64,
            application_requirements=[dict(id='official_top1_abs_gap_pp',comparison='<=',limit=1.)],
            approximation_requirements=[dict(id='max_absolute_error',comparison='<=',limit=.01)],whole_model_required=True)
        row=dict(original_reference=list(reference),declared_reference=list(reference),dsl_mathematical_output=list(reference),
            inside_frozen_domain=True,domain_sha256='d'*64,application_metrics=dict(official_top1_abs_gap_pp=.5),whole_model_output=True)
        self.assertTrue(observer._numerical_layers(policy,row,native)['numerical_observations_passed'])
        mutations=[('inside_frozen_domain',False),('domain_sha256','e'*64),('whole_model_output',False),
            ('application_metrics',dict(official_top1_abs_gap_pp=1.01)),
            ('dsl_mathematical_output',[reference[0]+1e-9]+reference[1:]),
            ('original_reference',[reference[0]+.1]+reference[1:])]
        for key,value in mutations:
            bad=copy.deepcopy(row);bad[key]=value
            with self.subTest(field=key):self.assertFalse(observer._numerical_layers(policy,bad,native)['numerical_observations_passed'])
        self.assertFalse(native['first_numeric_passed']);self.assertFalse(native['compiler_host_input_samekey_resource_or_first_delivery_verified'])


if __name__ == '__main__': unittest.main()
