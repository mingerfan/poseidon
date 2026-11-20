"""Twelve new data-reader controls. Fixtures are not compiler/HE executions."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from component_contract import prepare_task
from candidate_contract import request_input_names
from delivery_component import bound, canonical, sha
import poseidon_runtime_compile as entry
import compiler_plan_provenance as graph
import retained_compiler_provenance as reader


T = lambda c=2,s=45,l=12:'tensor<1x!ckks.poly<%d * %d * %d>>'%(c,s,l)
E = 'tensor<1x!earth.ci<0 * 0>>'


def fixture(alias=False):
    model=dict(format='poseidon-model-graph-v2',resource_profile='general-tensors-v1',id='compiler_fixture',
        inputs=[dict(name='input0',shape=[3]),dict(name='input1',shape=[3])],constants={},
        nodes=[dict(id='a',op='add',inputs=['input0','input1'],attrs={},outputs=['v0']),
               dict(id='b',op='multiply',inputs=['input0','input1'],attrs={},outputs=['v1'])],
        outputs=[dict(name='sum',value='v0'),dict(name='product',value='v1')])
    if alias:model['outputs'].append(dict(name='sum_alias',value='v0'))
    options=dict(chunk_period=4,compiler_configuration='poseidon-cpu-eva-w45-current-v1')
    if alias:options['output_binding_policy']='exact-values-v1'
    request=prepare_task(model,options);names=request_input_names(request)
    source=('@hc.func("'+','.join('c'for n in names)+'")\r\ndef golden('+', '.join(names)+'):\r\n    return ['+names[0]+'+'+names[1]+', '+names[0]+'*'+names[1]+']\r\n\r\n')
    frontend=('"builtin.module"() <{sym_name = "/out/candidate_trace.mlir"}> ({\n'
        '  "func.func"() <{function_type = ('+', '.join(E for _ in names)+') -> ('+E+', '+E+'), sym_name = "_hecate_golden"}> ({\n'
        '  ^bb0('+', '.join('%%arg%d: %s'%(i,E)for i in range(len(names)))+'):\n'
        '    %0 = "earth.add"(%arg0, %arg1) : ('+E+', '+E+') -> '+E+'\n'
        '    %1 = "earth.mul"(%arg0, %arg1) : ('+E+', '+E+') -> '+E+'\n'
        '    "func.return"(%0, %1) : ('+E+', '+E+') -> ()\n'
        '  }) : () -> ()\n}) : () -> ()\n')
    lowered=('module {\n  func.func @_hecate_golden('+', '.join('%%arg%d: %s'%(i,T())for i in range(len(names)))+') -> ('+T(2,30,11)+', '+T(2,30,11)+') attributes {arg_scale = array<i64: 45, 45, 45>} {\n'
        '    %0 = "ckks.encode"() <{payload = dense<[0.25, 0.5, 0.75, 1.0]> : tensor<4xf64>}> : () -> '+T(1)+'\n'
        '    %1 = "ckks.addcc"(%arg0, %arg1) : ('+T()+', '+T()+') -> '+T()+'\n'
        '    %2 = "ckks.mulcp"(%1, %0) : ('+T()+', '+T(1)+') -> '+T(2,90)+'\n'
        '    %3 = "ckks.rescalec"(%2) : ('+T(2,90)+') -> '+T(2,30,11)+'\n'
        '    %4 = "ckks.mulcc"(%arg0, %arg1) : ('+T()+', '+T()+') -> '+T(3,90)+'\n'
        '    %5 = "ckks.relinearize"(%4) : ('+T(3,90)+') -> '+T(2,90)+'\n'
        '    %6 = "ckks.rescalec"(%5) : ('+T(2,90)+') -> '+T(2,30,11)+'\n'
        '    return %3, %6 : '+T(2,30,11)+', '+T(2,30,11)+'\n  }\n}\n')
    command=entry.command(45,entry_policy=entry.ENTRY_POLICY,serialization_policy='runtime-plan-json-files-v1')
    ctx='poseidon-agent-n15-q13-p1-tc128-v1'
    value=lambda i,c,s,l:dict(id=str(i),kind='plaintext'if c==1 else'ciphertext',components=c,scale_log2=s,level=l,ntt=True,context=ctx,place=dict(kind='host',rank=0))
    raw=dict(format_version=1,plan_id='1',target=dict(target_id='poseidon-ckks-cpu',capability_version=1,
        operator_spec=dict(id='poseidon-agent-cpu-v1',version=1,source_sha256=next(x.split('=',1)[1]for x in command if x.startswith('--runtime-plan-operator-spec-sha256='))),
        world_size=1,device_counts=[0]),values=[value(i,2,45,12)for i in range(3)]+[value(3,1,45,12),value(4,2,45,12),
            value(5,2,90,12),value(6,2,30,11),value(7,3,90,12),value(8,2,90,12),value(9,2,30,11)],
        external_inputs=['0','1','2'],initialization=[dict(kind='encode',ordinal=0,output='3',payload=dict(kind='inline',values=[.25,.5,.75,1.]))],
        execution=[dict(kind='compute',op='add_cc',inputs=['0','1'],output='4',place=dict(kind='host',rank=0),ordinal=1),
            dict(kind='compute',op='mul_cp',inputs=['4','3'],output='5',place=dict(kind='host',rank=0),ordinal=2),
            dict(kind='compute',op='rescale',inputs=['5'],output='6',place=dict(kind='host',rank=0),ordinal=3,attrs=dict(target_level=11,target_scale_log2=30)),
            dict(kind='compute',op='mul_cc',inputs=['0','1'],output='7',place=dict(kind='host',rank=0),ordinal=4),
            dict(kind='compute',op='relinearize',inputs=['7'],output='8',place=dict(kind='host',rank=0),ordinal=5),
            dict(kind='compute',op='rescale',inputs=['8'],output='9',place=dict(kind='host',rank=0),ordinal=6,attrs=dict(target_level=11,target_scale_log2=30))],
        finalization=[],final_outputs=['6','9'])
    return request,source,frontend,lowered,raw,command


def payload(plan,op,which):
    values=op['payload']['values'];return struct.pack('<'+'d'*len(values),*values)


class Package:
    """Hand-built protocol fixture; the receipt explicitly grants no execution."""
    def __init__(self,root):
        self.root=root;root.mkdir();self.inventory={}
        self.request,self.source,self.frontend,self.lowered,self.raw,self.command=fixture()
        config=self.request['compiler_configuration'];self.compact,serialization=entry.compact_f64_trace(self.frontend)
        self.selected,self.selection=entry.entrypoint_trace(self.compact);self.selection['trace_serialization']=serialization
        self.packed=copy.deepcopy(self.raw)
        self.packed['initialization'][0]['payload']['values']*=4096
        self.put('request.json',canonical(self.request));self.put('attempt-00/candidate.py',self.source.encode())
        self.put('attempt-00/trace-payload.json',canonical(dict(request=self.request,candidate=dict(schema=1,request_id=self.request['request_id'],hecate_source=self.source))))
        self.put('attempt-00/output/trace-evidence.json',canonical(dict(frontend='real_Hecate',candidate_python_executed=False,request_id=self.request['request_id'])))
        for name,text in [('candidate_trace.mlir',self.frontend),('candidate_trace.compact.mlir',self.compact),('candidate_trace.entry.mlir',self.selected),('lowered.ckks.mlir',self.lowered)]:self.put('attempt-00/output/'+name,text.encode())
        for name,value in [('entry-selection.json',self.selection),('lowered._hecate_golden.runtime-plan.json',self.raw),('bound/plan.json',self.packed)]:self.put('attempt-00/output/'+name,canonical(value))
        self.report=dict(backend='poseidon_cpu',request_id=self.request['request_id'],synthetic_fixture=True,
            attempts=[dict(index=0,source_parsed=True,checked=True,compiled=True,compile_command=self.command,
                           runtime_plan_entry=self.selection,comparison=dict(reference=[123.456],actual=[987.654]))])
        self.put('report.json',canonical(self.report));self.proof_path=root.parent/(root.name+'-proof.json');self.seal()
    def put(self,name,raw):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);self.inventory[name]=dict(sha256=sha(raw),bytes=len(raw))
    def seal(self):
        self.proof=bound(dict(format='poseidon-first-source-qualification-binding-v1',evidence=str(self.root),files=self.inventory,
            origin='synthetic_fixture',request_id=self.request['request_id'],source_sha256=sha(self.source.encode()),candidate_preserved=True,
            compiler_input_bound=True,real_compilation_observed=True,compiler_input='/out/candidate_trace.entry.mlir',
            compiler_sha256='c'*64,backend_identity_sha256='b'*64))
        raw=canonical(self.proof);self.proof_path.write_bytes(raw);self.description=dict(path=str(self.proof_path),sha256=sha(raw))
    def inspect(self):return reader.inspect(self.description,self.request,self.source.encode(),maximum_total_read_bytes=8*1024**2)


class CompilerProvenanceControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.request,self.source,self.frontend,self.lowered,self.raw,self.command=fixture()
    def dag(self,text=None,raw=None):return graph.audit_dag(self.lowered if text is None else text,self.raw if raw is None else raw,self.command,payload)
    def test_01_metadata_projection_skips_numeric_bodies_and_rejects_ambiguous_keys(self):
        raw=b'{"attempts":[{"compile_command":["/hecate-opt","/out/input.mlir"],"comparison":{"actual":[1e99999,-2.5]}}]}'
        out=reader.project_json(raw,[('attempts',0,'compile_command')]);self.assertEqual(list(out.values()),[['/hecate-opt','/out/input.mlir']])
        for bad in [b'{"attempts":[],"attempts":[]}',b'{"x":[1,]}',b'{"x":NaN}',b'{} trailing']:
            with self.subTest(bad=bad),self.assertRaises(ValueError):reader.project_json(bad,[])
    def test_02_complete_typed_DAG_and_result_order_are_derived_from_SSA(self):
        d=self.dag();self.assertEqual([r['value_id']for r in d['parameters']],['0','1','2'])
        self.assertEqual([r['value_id']for r in d['returns']],['6','9']);self.assertEqual(d['instruction_count'],7)
        self.assertFalse(d['source_model_semantics_or_native_execution_certified'])
    def test_03_wrong_operands_types_schedule_and_final_order_are_rejected(self):
        for change in ('operand','type','phase','return'):
            p=copy.deepcopy(self.raw)
            if change=='operand':p['execution'][0]['inputs'].reverse()
            if change=='type':p['values'][6]['level']=10
            if change=='phase':p['execution'].insert(0,p['initialization'].pop())
            if change=='return':p['final_outputs'].reverse()
            with self.subTest(change=change),self.assertRaises(ValueError):self.dag(raw=p)
    def test_04_explicit_stable_ids_and_online_Encode_keep_exporter_numbering(self):
        text=self.lowered.replace('payload = dense<','runtime.value_id = 40 : i64, runtime.streaming, payload = dense<',1)
        p=copy.deepcopy(self.raw);ids={str(i):str(i if i<3 else 40 if i==3 else i-1)for i in range(10)}
        for value in p['values']:value['id']=ids[value['id']]
        p['values'].sort(key=lambda v:int(v['id']))
        for op in p['initialization']+p['execution']:
            op['output']=ids[op['output']]
            if 'inputs'in op:op['inputs']=[ids[n]for n in op['inputs']]
        p['final_outputs']=[ids[n]for n in p['final_outputs']];p['execution'].insert(0,p['initialization'].pop())
        d=self.dag(text,p);self.assertEqual([r['value_id']for r in d['returns']],['5','8'])
    def test_05_public_payload_bit_identity_and_periodic_binding_are_independent(self):
        a=graph.dense('dense<[0.25, 0.5]> : tensor<2xf64>')
        self.assertEqual(a,graph.dense('dense<"0x'+a.hex()+'"> : tensor<2xf64>'))
        self.assertEqual(graph.dense('dense<-0.0> : tensor<2xf64>'),struct.pack('<dd',-0.,-0.))
        packed=copy.deepcopy(self.raw);packed['initialization'][0]['payload']['values']*=4096
        result=graph.audit_periodic_binding(self.raw,packed,4,payload);self.assertFalse(result['plaintext_values_changed'])
        packed['initialization'][0]['payload']['values'][-1]+=.01
        with self.assertRaises(ValueError):graph.audit_periodic_binding(self.raw,packed,4,payload)
        packed=copy.deepcopy(self.raw);packed['execution'][0]['op']='mul_cc'
        with self.assertRaises(ValueError):graph.audit_periodic_binding(self.raw,packed,4,payload)
    def test_06_multitensor_parameters_and_multiple_named_results_preserve_order(self):
        _,selected=entry.entrypoint_trace(self.frontend)
        mapping=graph.layout_mapping(self.request,self.source,self.frontend,self.dag())
        self.assertEqual(len(mapping['inputs']),3);self.assertEqual([o['name']for o in mapping['outputs']],['sum','product'])
        self.assertEqual(mapping['logical_values'],6)
        wrong=self.source.replace('golden(x0, x1, zero_ct)','golden(x1, x0, zero_ct)')
        with self.assertRaises(ValueError):graph.layout_mapping(self.request,wrong,self.frontend,self.dag())
    def test_07_exact_aliases_count_all_logical_cells_without_duplicate_physical_returns(self):
        request,source,front,low,raw,command=fixture(alias=True)
        d=graph.audit_dag(low,raw,command,payload);m=graph.layout_mapping(request,source,front,d)
        self.assertEqual(m['logical_values'],9);self.assertEqual(m['unique_physical_cells'],6);self.assertEqual(m['alias_cells'],3)
        self.assertEqual(m['outputs'][0]['segments'],m['outputs'][2]['segments'])
        self.assertEqual(m['outputs'][2]['alias_of'],'sum')
        model=copy.deepcopy(request['model']);model['inputs'][0]['shape']=[6];model['inputs'][1]['shape']=[6];model['outputs']=model['outputs'][:2]
        larger=prepare_task(model,dict(chunk_period=4,compiler_configuration='poseidon-cpu-eva-w45-current-v1'))
        names=request_input_names(larger);large_source='@hc.func("'+','.join('c'for n in names)+'")\ndef golden('+', '.join(names)+'):\n    return ['+', '.join(names[:4])+']\n'
        large_front=('  "func.func"() <{sym_name = "_hecate_golden"}> ({\n  ^bb0('+', '.join('%%arg%d: %s'%(i,E)for i in range(len(names)))+'):\n'
                     '    "func.return"(%0, %1, %2, %3) : () -> ()\n  }) : () -> ()\n')
        larger_dag=bound(dict(parameters=[dict(ssa='%arg'+str(i),value_id=str(i),type=dict(kind='ciphertext'))for i in range(len(names))],
            returns=[dict(ssa='%'+str(i),value_id=str(i+len(names)),type=dict(kind='ciphertext'))for i in range(4)]))
        large_map=graph.layout_mapping(larger,large_source,large_front,larger_dag)
        self.assertEqual(len(large_map['inputs']),5);self.assertEqual(large_map['logical_values'],12)
        self.assertEqual([(o['segments'][-1]['logical_start'],o['segments'][-1]['elements'])for o in large_map['outputs']],[(4,2),(4,2)])
    def test_08_no_guessed_placement_unknown_attributes_or_physical_aliases(self):
        with self.assertRaises(ValueError):self.dag(self.lowered.replace('"ckks.addcc"(%arg0, %arg1)','"ckks.addcc"(%arg0, %arg1) <{new_semantics = 1 : i64}>'))
        with self.assertRaises(ValueError):self.dag(self.lowered.replace('arg_scale =','dist.device_counts = array<i64: 1>, arg_scale ='))
        d=self.dag();d['returns'][1]['value_id']=d['returns'][0]['value_id'];d=bound({k:v for k,v in d.items()if k!='binding'})
        with self.assertRaises(ValueError):graph.layout_mapping(self.request,self.source,self.frontend,d)
    def test_09_files_are_bounded_regular_distinct_and_cannot_escape_root(self):
        root=self.root/'files';root.mkdir();a=root/'a.json';a.write_bytes(b'{}');inventory={'a.json':dict(sha256=sha(b'{}'),bytes=2)}
        f=reader.Files(root,inventory,maximum_total=2);self.assertEqual(f.read('a.json'),b'{}')
        for name in ('../a.json','/tmp/a.json','.env','execution.npy'):
            with self.subTest(name=name),self.assertRaises(ValueError):f.read(name)
        a.write_bytes(b'[]')
        with self.assertRaises(ValueError):f.verify()
    def test_10_frozen_file_pipeline_preserves_scope_and_never_opens_numeric_outputs(self):
        p=Package(self.root/'pipeline');result=p.inspect()
        self.assertEqual(result['logical_IO']['logical_values'],6);self.assertEqual(result['DAG']['instruction_count'],7)
        self.assertFalse(result['compiler_executed']);self.assertFalse(result['complex_program_eligibility_or_first_delivery_success_granted'])
        self.assertEqual(result['numerical_output_or_reference_values_decoded'],0)
        self.assertTrue(all(not any(t in name for t in ('decrypted','execution.json','inputs.json','arrays.npz'))for name in result['files_read']))
    def test_11_valid_hashes_do_not_hide_modified_entry_compiler_command_or_frontend_DSL(self):
        for i,change in enumerate(('entry','command','payload')):
            p=Package(self.root/('bad'+str(i)))
            if change=='entry':p.put('attempt-00/output/candidate_trace.entry.mlir',p.selected.replace('earth.add','earth.mul').encode())
            if change=='command':
                p.report['attempts'][0]['compile_command']=p.command+['--unapproved-option'];p.put('report.json',canonical(p.report))
            if change=='payload':
                v=json.loads((p.root/'attempt-00/trace-payload.json').read_bytes());v['candidate']['hecate_source']+='\n';p.put('attempt-00/trace-payload.json',canonical(v))
            p.seal()
            with self.subTest(change=change),self.assertRaises(ValueError):p.inspect()
    def test_12_context_payload_or_ir_tamper_cannot_be_rebound_as_the_original_program(self):
        for i,change in enumerate(('context','payload','DAG')):
            p=Package(self.root/('wrong'+str(i)))
            if change=='context':p.raw['values'][0]['context']='other-context';name='lowered._hecate_golden.runtime-plan.json';value=p.raw
            elif change=='payload':p.packed['initialization'][0]['payload']['values'][0]=-.25;name='bound/plan.json';value=p.packed
            else:p.raw['execution'][0]['inputs'].reverse();name='lowered._hecate_golden.runtime-plan.json';value=p.raw
            p.put('attempt-00/output/'+name,canonical(value));p.seal()
            with self.subTest(change=change),self.assertRaises(ValueError):p.inspect()


if __name__=='__main__':unittest.main()
