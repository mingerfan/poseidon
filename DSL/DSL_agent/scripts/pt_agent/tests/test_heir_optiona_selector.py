"""Small deterministic checks for the explicit exact OptionA representation."""
import os
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pt_agent import paths
from pt_agent.tests._artifacts import require


SYNTAX_FIXTURE = '''module {
  func.func @logic(%x: memref<1xf64>, %value: i32) -> memref<1xf64> attributes {note = "%c-1_i32"} {
    %__native_alpha_0 = arith.constant 0 : index
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %one = arith.constant 1.0 : f64
    %two = arith.constant 2.0 : f64
    %three = arith.constant 3.0 : f64
    %lower = arith.cmpi sge, %value, %c-1_i32 : i32
    %upper = arith.cmpi sle, %value, %c1 : i32
    %both = arith.andi %lower, %upper : i1
    scf.if %both {
      %zero = arith.cmpi eq, %value, %c0 : i32
      %positive = arith.cmpi eq, %value, %c1 : i32
      %either = arith.ori %zero, %positive : i1
      scf.if %either {
        memref.store %one, %x[%__native_alpha_0] : memref<1xf64>
      } else {
        memref.store %two, %x[%__native_alpha_0] : memref<1xf64>
      }
    } else {
      memref.store %three, %x[%__native_alpha_0] : memref<1xf64>
    }
    return %x : memref<1xf64>
  }
  func.func @fresh(%evaluator: !evaluator, %x: !ct, %step: index) -> !ct {
    %rotated = lattigo.ckks.rotate_new %evaluator, %x, %step : (!evaluator, !ct, index) -> !ct
    return %rotated : !ct
  }
}
'''


class NativeSyntaxChecks(unittest.TestCase):
    def test_negative_names_are_injective_and_quoted_attributes_are_preserved(self):
        from pt_agent.experiments.heir_native_clear_ir import canonical_ssa_names,parse
        text,mapping=canonical_ssa_names(SYNTAX_FIXTURE)
        self.assertEqual(mapping,{'%c-1_i32':'%__native_alpha_1'})
        self.assertIn('note = "%c-1_i32"',text)
        self.assertEqual(text.count('\n'),SYNTAX_FIXTURE.count('\n'))
        self.assertEqual(parse(SYNTAX_FIXTURE).ssa_alpha_renames,mapping)

    def test_nested_else_bodies_are_retained_in_the_operation_walk(self):
        from pt_agent.experiments.heir_native_clear_ir import parse,walk
        program=parse(SYNTAX_FIXTURE)
        nodes=list(walk(program.functions['logic'].body))
        self.assertEqual(sum(n.kind=='memref.store' for n in nodes),3)
        self.assertEqual(sum(n.kind=='if' and 'else_body' in n.attrs for n in nodes),2)


def selector_fixture():
    import numpy as np
    shape={'x':(1,2,4,6),'conv':(1,2,4,6),'act':(1,2,4,6),
           'rows':(1,2,2,6),'cols':(1,2,2,3),'out':(1,5,2,3)}
    def node(id,op,source,target,attrs):return SimpleNamespace(id=id,op=op,inputs=[source],outputs=[target],attrs=attrs,src='synthetic-selector')
    nodes=[node('n0','conv2d','x','conv',dict(weight='original.weight',stride=(1,1),padding=(0,0),dilation=(1,1),groups=1)),
           node('n1','act','conv','act',dict(fn='silu')),
           node('n2','slice','act','rows',dict(axis=2,start=0,end=4,step=2)),
           node('n3','slice','rows','cols',dict(axis=3,start=0,end=6,step=2)),
           node('n4','pad','cols','out',dict(pads=(0,0,0,0,1,2)))]
    graph=SimpleNamespace(name='selector-small',nodes=nodes,
        tensors={name:SimpleNamespace(name=name,shape=value,kind='features') for name,value in shape.items()},
        weights={'original.weight':np.array([[[[.5]],[[.25]]],[[[-.75]],[[1.25]]]],dtype=np.float64)},inputs=['x'],outputs=['out'])
    program='@fhe.program(slots=16384)\ndef small(x: Cipher[1,2,4,6]):\n    y = act(x, fn="silu", method="cheb", deg=63, domain=(-4.0,4.0))\n    return y\n'
    return graph,program


class SelectorAdapterChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
        import numpy as np
        cls.np=np

    def test_two_actual_shapes_and_rectangular_shape_have_independent_strict_math(self):
        from pt_agent.experiments.heir_cnn_export import optiona_selector_weights
        np=self.np;total=0
        for channels,height,width,low,high in [(16,32,32,8,8),(32,16,16,16,16),(2,4,6,1,2)]:
            x=(np.arange(channels*height*width,dtype=np.float64).reshape(channels,height,width)-11.5)/7
            weights=optiona_selector_weights(channels,low,high)
            observed=np.zeros((channels+low+high,height//2,width//2),dtype=np.float64)
            for oc in range(observed.shape[0]):
                for h in range(observed.shape[1]):
                    for w in range(observed.shape[2]):
                        value=0.
                        for ic in range(channels):value=float(value+float(x[ic,2*h,2*w]*weights[oc,ic,0,0]))
                        observed[oc,h,w]=value
            reference=np.pad(x[:,::2,::2],((low,high),(0,0),(0,0)))
            self.assertTrue(np.isfinite(observed).all());self.assertEqual(observed.dtype,np.float64)
            self.assertTrue(np.all(np.abs(observed-reference)<=1e-12+1e-12*np.abs(reference)))
            self.assertEqual(float(np.abs(observed-reference).max()),0.)
            self.assertFalse(weights.flags.writeable);total+=observed.size
        self.assertEqual(total,12318)

    def test_default_keeps_original_slices_and_explicit_option_preserves_graph_identity(self):
        from pt_agent.experiments.heir_cnn_export import export_graph
        graph,program=selector_fixture();original=graph.weights['original.weight'].copy()
        default,old=export_graph(graph,program,function_name='small')
        selected,new=export_graph(graph,program,function_name='small',exact_optiona_selector=True)
        self.assertEqual(default.count('tensor.extract_slice'),2);self.assertEqual(default.count('tensor.pad'),1)
        self.assertEqual(default.count('linalg.conv_2d_nchw_fchw'),1);self.assertFalse(old['exact_optiona_selector'])
        self.assertEqual(selected.count('tensor.extract_slice'),0);self.assertEqual(selected.count('tensor.pad'),0)
        self.assertEqual(selected.count('linalg.conv_2d_nchw_fchw'),2)
        self.assertEqual(new['graph_nodes'],5);self.assertEqual(len(new['graph_operations']),5)
        self.assertEqual(new['exact_optiona_lowerings'][0]['original_nodes'],['n2','n3','n4'])
        self.assertEqual(new['exact_optiona_lowerings'][0]['output'],'out')
        self.assertEqual(new['output_shape'],[1,5,2,3]);self.assertEqual(new['approximations'][0]['deg'],63)
        self.assertTrue(self.np.array_equal(original,graph.weights['original.weight']))
        self.assertEqual(set(graph.weights),{'original.weight'})

    def test_shared_or_reported_intermediates_are_rejected(self):
        from pt_agent.experiments.heir_cnn_export import exact_optiona_selectors
        graph,_=selector_fixture();graph.outputs.append('rows')
        with self.assertRaisesRegex(ValueError,'another consumer/output'):exact_optiona_selectors(graph)
        graph,_=selector_fixture();graph.nodes.append(SimpleNamespace(id='extra',op='add',inputs=['rows','rows'],outputs=['extra'],attrs={}))
        with self.assertRaisesRegex(ValueError,'another consumer/output'):exact_optiona_selectors(graph)

    def test_changed_axes_ranges_padding_or_shapes_are_rejected(self):
        from pt_agent.experiments.heir_cnn_export import exact_optiona_selectors
        changes=[('axis',lambda g:g.nodes[2].attrs.update(axis=3)),
                 ('range',lambda g:g.nodes[3].attrs.update(start=1)),
                 ('value',lambda g:g.nodes[4].attrs.update(value=.5)),
                 ('mode',lambda g:g.nodes[4].attrs.update(mode='reflect')),
                 ('spatialpad',lambda g:g.nodes[4].attrs.update(pads=(1,0,0,0,1,2))),
                 ('shape',lambda g:setattr(g.tensors['rows'],'shape',(1,2,2,5)))]
        for name,change in changes:
            with self.subTest(name=name):
                graph,_=selector_fixture();change(graph)
                with self.assertRaises(ValueError):exact_optiona_selectors(graph)

    def test_constant_inventory_separates_public_selector_and_detects_changed_coefficients(self):
        import re
        from pt_agent.experiments.heir_cnn_export import export_graph
        from pt_agent.experiments.heir_cnn_math_check import check_constants
        graph,program=selector_fixture();text,_=export_graph(graph,program,exact_optiona_selector=True)
        rows=check_constants(graph,text,exact_optiona_selector=True)
        self.assertEqual([r['name'] for r in rows],['original.weight','derived-exact-optiona:n4'])
        self.assertTrue(all(r['numerically_equal'] and r['passed'] for r in rows))
        match=list(re.finditer(r'^\s*%\w+ = arith.constant dense<(.+)> : tensor<[^>]+>$',text,re.MULTILINE))[-1]
        literal=match[1].replace('1.00000000000000000e+00','5.00000000000000000e-01',1)
        changed=text[:match.start(1)]+literal+text[match.end(1):]
        with self.assertRaisesRegex(ValueError,'derived-exact-optiona'):check_constants(graph,changed,exact_optiona_selector=True)


class SpatialMeanAdapterChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
        import numpy as np
        cls.np=np

    def fixture(self):
        graph,program=selector_fixture();np=self.np
        graph.nodes.extend([SimpleNamespace(id='n5',op='mean',inputs=['out'],outputs=['pooled'],attrs=dict(axes=(2,3))),
                            SimpleNamespace(id='n6',op='linear',inputs=['pooled'],outputs=['logits'],attrs=dict(weight='head.weight',bias='head.bias'))])
        graph.tensors['pooled']=SimpleNamespace(shape=(1,5));graph.tensors['logits']=SimpleNamespace(shape=(1,3));graph.outputs=['logits']
        graph.weights.update({'head.weight':np.arange(15,dtype=np.float64).reshape(3,5)/16,'head.bias':np.array([1.,-2.,3.])})
        return graph,program

    def test_all_public_coefficients_and_channel_ownership_for_real_and_rectangular_shapes(self):
        from pt_agent.experiments.heir_cnn_export import spatial_mean_matrix
        np=self.np
        for shape in ((1,2,2,4),(1,3,2,6),(1,64,8,8)):
            with self.subTest(shape=shape):
                _,C,H,W=shape;matrix=spatial_mean_matrix(shape);expected=np.zeros((C,C*H*W))
                for c in range(C):expected[c,c*H*W:(c+1)*H*W]=1/(H*W)
                self.assertEqual(matrix.dtype,np.float64);self.assertFalse(matrix.flags.writeable)
                self.assertTrue(np.array_equal(matrix,expected));self.assertEqual(np.count_nonzero(matrix),C*H*W)

    def test_signed_coordinate_math_matches_original_spatial_sum_and_factor(self):
        from pt_agent.experiments.heir_cnn_export import spatial_mean_matrix
        from pt_agent.experiments.heir_cnn_math_check import check_math
        np=self.np
        for shape in ((1,2,2,4),(1,3,2,6),(1,64,8,8)):
            _,C,H,W=shape;values=(np.arange(C*H*W,dtype=np.float64)-C*H*W/2).reshape(shape)
            original=values.reshape(1,C,-1).sum(-1)*(1/(H*W));observed=(spatial_mean_matrix(shape)@values.reshape(-1))[None,:]
            self.assertTrue(check_math(original,observed)['passed'])

    def test_option_keeps_original_nodes_weights_coefficients_and_old_default(self):
        from pt_agent.experiments.heir_cnn_export import export_graph
        graph,program=self.fixture();before={k:v.copy() for k,v in graph.weights.items()}
        old,original=export_graph(graph,program);new,changed=export_graph(graph,program,exact_optiona_selector=True,exact_spatial_mean_matvec=True)
        self.assertFalse(original['exact_spatial_mean_matvec']);self.assertEqual(original['exact_spatial_mean_lowerings'],[])
        self.assertEqual(original['graph_nodes'],changed['graph_nodes']);self.assertEqual(changed['graph_nodes'],7)
        self.assertEqual(original['approximations'],changed['approximations']);self.assertEqual(old.count('linalg.matvec'),1);self.assertEqual(new.count('linalg.matvec'),2)
        self.assertEqual(changed['exact_spatial_mean_lowerings'][0]['original_node'],'n5')
        self.assertFalse(changed['exact_spatial_mean_lowerings'][0]['trainable_weights_changed'])
        self.assertEqual(list(graph.weights),list(before))
        for key in before:self.assertTrue(self.np.array_equal(graph.weights[key],before[key]))

    def test_constant_inventory_names_public_pool_separately_and_rejects_changed_values(self):
        from pt_agent.experiments.heir_cnn_export import export_graph
        from pt_agent.experiments.heir_cnn_math_check import check_constants
        graph,program=self.fixture();text,_=export_graph(graph,program,exact_optiona_selector=True,exact_spatial_mean_matvec=True)
        rows=check_constants(graph,text,exact_optiona_selector=True,exact_spatial_mean_matvec=True)
        self.assertEqual([r['name'] for r in rows],['original.weight','derived-exact-optiona:n4','derived-exact-spatial-mean:n5','head.weight','head.bias'])
        self.assertTrue(all(r['passed'] and r['numerically_equal'] for r in rows))
        import re
        literals=list(re.finditer(r'^\s*%\w+ = arith.constant dense<(.+)> : tensor<[^>]+>$',text,re.MULTILINE));match=literals[2]
        changed=match[1].replace('1.66666666666666657e-01','5.00000000000000000e-01',1)
        self.assertNotEqual(changed,match[1])
        with self.assertRaisesRegex(ValueError,'derived-exact-spatial-mean'):
            check_constants(graph,text[:match.start(1)]+changed+text[match.end(1):],exact_optiona_selector=True,exact_spatial_mean_matvec=True)

    def test_unsupported_batch_shape_axes_and_nonboolean_option_are_rejected(self):
        from pt_agent.experiments.heir_cnn_export import spatial_mean_matrix,Emitter,Tensor,export_graph
        for shape in ((2,3,4,5),(1,3,0,5),(1,3,4),(-1,3,4,5)):
            with self.subTest(shape=shape),self.assertRaises(ValueError):spatial_mean_matrix(shape)
        with self.assertRaises(ValueError):Emitter().spatial_mean_matvec(Tensor('%x',(1,3,4,5)),(1,2))
        graph,program=self.fixture()
        with self.assertRaises(ValueError):export_graph(graph,program,exact_spatial_mean_matvec=1)


class SourceGateChecks(unittest.TestCase):
    def fixture(self,root):
        source={key:str(root/key) for key in ('mlir','graph','program')}
        paths_=[root/('proof'+str(i)+'.json') for i in range(3)]
        minimal=dict(status='ok',worker=dict(status='ok',coordinate_pairs=16384,coordinate_inputs_completed=2,actual_crypto_operations=0,
                    samples=[dict(passed=True,input_ownership_preserved=True,pairs=8192) for _ in range(2)]))
        paths_[0].write_text(json.dumps(minimal));paths_[1].write_text(json.dumps(dict(status='ok',tests=dict(count=5,skipped=False))))
        constants=[dict(name='original:'+str(i),passed=True,numerically_equal=True) for i in range(59)]
        constants +=[dict(name='derived-exact-optiona:'+str(i),passed=True,numerically_equal=True) for i in range(2)]
        exported=dict(status='ok',model='resnet20_silu',exact_optiona_selector=True,export_mlir=str(root/'new.mlir'),
            original_frontend=source['mlir'],graph_dir=source['graph'],program=source['program'],proofs=[str(p) for p in paths_],
            export_record=dict(exact_optiona_selector=True,graph_nodes=74,graph_operations=[{}]*74,approximations=[{}]*19,
                exact_optiona_lowerings=[dict(original_nodes=ids,trainable_weights_changed=False) for ids in
                                        (['n29','n30','n31'],['n53','n54','n55'])]),
            source_checks=dict(status='ok',same_saved_approximations=19,original_constants=59,derived_fixed_selector_constants=2,
                               reference_changes=0,tolerance_changes=0,constants=constants))
        record=root/'export.json';record.write_text(json.dumps(exported))
        native=dict(source_export_result=str(record),source=exported['export_mlir'])
        return source,native,record,exported

    def test_exact_selector_reuses_only_proved_unchanged_original_references(self):
        from pt_agent.experiments.heir_native_clear_math import checked_exact_optiona_source
        with tempfile.TemporaryDirectory() as name:
            source,native,_,_=self.fixture(Path(name));checked=checked_exact_optiona_source(native,source)
            self.assertEqual(checked['original_constants'],59);self.assertEqual(checked['saved_approximation_coefficients_preserved'],19)
            self.assertTrue(checked['original_reference_reused_by_exact_operator_equivalence'])

    def test_changed_original_constants_or_triples_are_rejected(self):
        from pt_agent.experiments.heir_native_clear_math import checked_exact_optiona_source
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for kind in ('constant','triple','reference'):
                source,native,path,exported=self.fixture(root)
                if kind=='constant':exported['source_checks']['constants'][0]['numerically_equal']=False
                elif kind=='triple':exported['export_record']['exact_optiona_lowerings'][0]['original_nodes']=['n28','n30','n31']
                else:exported['original_frontend']=str(root/'changed.mlir')
                path.write_text(json.dumps(exported))
                with self.subTest(kind=kind),self.assertRaises(ValueError):checked_exact_optiona_source(native,source)

    def terminal_fixture(self,root):
        original=root/'original';original.mkdir();ir=original/'model-lattigo.mlir';ir.write_text('module {}\n')
        raw=dict(status='error',model='resnet20_silu',backend_executed=False,guard_termination=dict(reason='rss_metadata_missing'),
            attempts=dict(compile=1),completed=dict(compile=0),processes=[dict(observation=dict(exit_code=0,timed_out=False,signals_sent=[],remaining_pids=[],signal=15))])
        raw_path=original/'result.json';raw_path.write_text(json.dumps(raw))
        audit=dict(status='ok',mode='independent-native-terminal-audit',model='resnet20_silu',native_compile_process_passed=True,
            original_complete_case_passed=False,compiler_runs=0,original_result_preserved=True,original_native_result=str(raw_path),
            compiled_artifact=str(ir),checks={key:True for key in ('native_ir_complete','same_native_parameters','source_shape_identity','no_child_termination_signal')})
        audit_path=root/'audit.json';audit_path.write_text(json.dumps(audit))
        return audit_path,raw_path,raw,ir

    def test_terminal_audit_resolves_exit_zero_artifact_and_keeps_original_rejection(self):
        from pt_agent.experiments.heir_native_clear_math import checked_native_source
        with tempfile.TemporaryDirectory() as name:
            path,original,raw,ir=self.terminal_fixture(Path(name));native,actual=checked_native_source(path)
            self.assertEqual(actual,ir);self.assertEqual(native['status'],'error');self.assertEqual(native['completed']['compile'],0)
            self.assertEqual(json.loads(original.read_text()),raw)

    def test_terminal_audit_rejects_child_failure_or_signal_and_new_compile(self):
        from pt_agent.experiments.heir_native_clear_math import checked_native_source
        for kind in ('exit','signal','timeout','new_compile'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as name:
                path,original,raw,_=self.terminal_fixture(Path(name))
                if kind=='exit':raw['processes'][0]['observation']['exit_code']=1
                elif kind=='signal':raw['processes'][0]['observation']['signals_sent']=[dict(signal=15)]
                elif kind=='timeout':raw['processes'][0]['observation']['timed_out']=True
                else:
                    audit=json.loads(path.read_text());audit['compiler_runs']=1;path.write_text(json.dumps(audit))
                original.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):checked_native_source(path)


class FixedPoolSourceGateChecks(unittest.TestCase):
    def fixture(self,root):
        source,native,path,exported=SourceGateChecks().fixture(root)
        native['exact_spatial_mean_matvec']=True;exported['exact_spatial_mean_matvec']=True
        exported['export_record'].update(exact_spatial_mean_matvec=True,exact_spatial_mean_lowerings=[dict(original_node='n72',input_shape=[1,64,8,8],
            output_shape=[1,64],axes=[2,3],matrix_shape=[64,4096],coefficient=1/64,nonzeros=4096,trainable_weights_changed=False,bias=False)])
        exported['source_checks']['derived_fixed_pool_constants']=1
        exported['source_checks']['constants'].append(dict(name='derived-exact-spatial-mean:n72',passed=True,numerically_equal=True))
        p=root/'pool-tests.json';p.write_text(json.dumps(dict(status='ok',tests=dict(count=5,skipped=False))))
        q=root/'pool-native.json';q.write_text(json.dumps(dict(status='ok',worker=dict(status='ok',math_pairs=30,actual_crypto_operations=0,actual_model_forwards=0,
            samples=[dict(passed=True,pairs=10,actual_client_pack_verified=True,input_ownership_preserved=True) for _ in range(3)]))))
        exported['fixed_pool_proofs']=[str(p),str(q)];path.write_text(json.dumps(exported));return source,native,path,exported

    def test_original_reference_is_reused_only_with_separate_fixed_pool_proofs(self):
        from pt_agent.experiments.heir_native_clear_math import checked_exact_optiona_source
        with tempfile.TemporaryDirectory() as name:
            source,native,_,_=self.fixture(Path(name));record=checked_exact_optiona_source(native,source)
            self.assertEqual(record['fixed_spatial_mean']['native_pairs'],30)
            self.assertEqual(record['fixed_spatial_mean']['original_node'],'n72');self.assertEqual(record['reference_changes'],0)

    def test_wrong_pool_shape_coefficient_or_failed_native_proof_is_rejected(self):
        from pt_agent.experiments.heir_native_clear_math import checked_exact_optiona_source
        for kind in ('shape','coefficient','proof','explicit_option'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as name:
                source,native,path,exported=self.fixture(Path(name))
                if kind=='shape':exported['export_record']['exact_spatial_mean_lowerings'][0]['input_shape']=[1,64,8,9]
                elif kind=='coefficient':exported['export_record']['exact_spatial_mean_lowerings'][0]['coefficient']=1/32
                elif kind=='explicit_option':native['exact_spatial_mean_matvec']=False
                else:
                    q=Path(exported['fixed_pool_proofs'][1]);data=json.loads(q.read_text());data['worker']['samples'][0]['passed']=False;q.write_text(json.dumps(data))
                path.write_text(json.dumps(exported))
                with self.assertRaises(ValueError):checked_exact_optiona_source(native,source)


GO=Path(os.environ.get('PT_HEIR_GO_BINARY',str(paths.WORKSPACE/'baselines/heir-r20-go-backend-20261115/toolchain/go/bin/go')))


@require(GO)
class NativeRuntimeChecks(unittest.TestCase):
    def test_native_boolean_else_and_fresh_rotation_have_actual_small_math(self):
        from pt_agent.experiments.heir_native_clear_ir import parse
        from pt_agent.experiments.heir_native_clear_codegen import Emitter,RUNTIME
        from pt_agent.experiments.r10_paper_metrics import run_process
        from pt_agent.experiments.heir_native_clear_math import environment
        program=parse(SYNTAX_FIXTURE);emitter=Emitter(program,[],prove=False)
        source=RUNTIME+'\n'+'\n'.join(emitter.function(f) for f in program.functions.values())
        source+='''
func main(){runtime.GOMAXPROCS(1);debug.SetMemoryLimit(64<<20);_=syscall.Getrusage
 for _,v:=range []int64{-2,-1,0,1,2}{t:=splat([]int64{1},0);got:=f_logic(t,v).get(0);want:=3.0;if v==-1{want=2};if v==0||v==1{want=1};if got!=want{panic("boolean/else math")}}
 a:=&V{data:[]float64{1,2,3,4},level:3};b:=f_fresh(nil,a,1);for i,want:=range []float64{2,3,4,1}{if b.data[i]!=want{panic("rotation direction")}};b.data[0]=99;if a.data[0]!=1||a.data[1]!=2{panic("fresh rotation ownership")}
 fmt.Println("native boolean/else/rotation math passed")
}
'''
        with tempfile.TemporaryDirectory(prefix='heir-optiona-syntax-') as name:
            work=Path(name);(work/'main.go').write_text(source);(work/'go.mod').write_text('module optiona-small-math\n\ngo 1.26\n')
            env=environment(work,64)
            for stage,argv,seconds in [('build',[str(GO),'build','-p=1','-buildvcs=false','-gcflags=-l','-o',str(work/'small'),'.'],120),('math',[str(work/'small')],30)]:
                observation=run_process(argv,cwd=work,log=work/(stage+'.log'),resources=work/(stage+'-resources.jsonl'),timeout=seconds,mem_gib=2,env=env)
                self.assertEqual(observation['exit_code'],0,(work/(stage+'.log')).read_text())
                self.assertFalse(observation['timed_out']);self.assertFalse(observation['remaining_pids'])


if __name__=='__main__':unittest.main()
