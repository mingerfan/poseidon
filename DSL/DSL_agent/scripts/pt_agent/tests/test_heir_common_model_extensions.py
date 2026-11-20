"""Small source and independent arithmetic checks; no model weights or native tools."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


def pool_fixture(model='alexnet_silu_avgpool',kernel=3,padding=1,height=5,width=7):
    oh=(height+2*padding-kernel)//2+1;ow=(width+2*padding-kernel)//2+1
    shapes={'x':(1,2,height,width),'pool':(1,2,oh,ow),'out':(1,2*oh*ow)}
    nodes=[NS(id='pool',op='avgpool',inputs=['x'],outputs=['pool'],attrs=dict(kernel=[kernel,kernel],stride=[2,2],padding=[padding,padding]),src='small'),
           NS(id='reshape',op='reshape',inputs=['pool'],outputs=['out'],attrs=dict(shape=list(shapes['out'])),src='small')]
    return NS(name=model,inputs=['x'],outputs=['out'],nodes=nodes,
              tensors={n:NS(shape=s) for n,s in shapes.items()},weights={})


def export_fixture(graph):
    from pt_agent.experiments.heir_cnn_export import export_graph
    # Pool/reshape source fixture: no activation approximation is fitted.
    program=NS(params=[NS(shape=graph.tensors['x'].shape)])
    with patch('pt_agent.experiments.heir_cnn_export.saved_activations',return_value=(program,[])):
        return export_graph(graph,'small pool fixture',common_model=graph.name)


def scalar_pool(x,kernel,padding):
    import numpy as np
    n,c,h,w=x.shape;oh=(h+2*padding-kernel)//2+1;ow=(w+2*padding-kernel)//2+1
    out=np.zeros((n,c,oh,ow),dtype=np.float64)
    for channel in range(c):
        for row in range(oh):
            for col in range(ow):
                values=[]
                for kh in range(kernel):
                    for kw in range(kernel):
                        ih,iw=row*2+kh-padding,col*2+kw-padding
                        if 0<=ih<h and 0<=iw<w:values.append(float(x[0,channel,ih,iw]))
                out[0,channel,row,col]=sum(values)/(kernel*kernel)
    return out


def coefficient_convolution(x,weight,padding):
    import numpy as np
    kernel=weight.shape[-1];oh=(x.shape[2]+2*padding-kernel)//2+1;ow=(x.shape[3]+2*padding-kernel)//2+1
    out=np.zeros((1,weight.shape[0],oh,ow),dtype=np.float64)
    for oc in range(weight.shape[0]):
        for h in range(oh):
            for w in range(ow):
                for ic in range(weight.shape[1]):
                    for kh in range(kernel):
                        for kw in range(kernel):
                            ih,iw=h*2+kh-padding,w*2+kw-padding
                            if 0<=ih<x.shape[2] and 0<=iw<x.shape[3]:out[0,oc,h,w]+=x[0,ic,ih,iw]*weight[oc,ic,kh,kw]
    return out


class CommonArithmetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'

    def assert_math(self,reference,observed):
        from pt_agent.experiments.heir_cnn_math_check import check_math
        self.assertTrue(check_math(reference,observed)['passed'])

    def test_pool_coefficients_preserve_signed_channels_and_non_square_windows(self):
        import numpy as np
        from pt_agent.experiments.heir_common_models import avgpool_weights
        from pt_agent.experiments.heir_cnn_math_check import evaluate_graph
        for model,kernel,padding in [('vgg16_silu_avgpool',2,0),('alexnet_silu_avgpool',3,1)]:
            with self.subTest(model=model):
                graph=pool_fixture(model,kernel,padding)
                x=(np.arange(70,dtype=np.float64).reshape(1,2,5,7)-36)/8;x[:,1]*=-3
                weight,spec=avgpool_weights(x.shape,graph.nodes[0].attrs,graph.tensors['pool'].shape)
                self.assertFalse(weight.flags.writeable)
                self.assertEqual(spec['divisor'],kernel*kernel)
                self.assertEqual(int(np.count_nonzero(weight)),2*kernel*kernel)
                reference=scalar_pool(x,kernel,padding)
                self.assert_math(reference,coefficient_convolution(x,weight,padding))
                observed,_=evaluate_graph(graph,[],x,common_model=model)
                self.assert_math(reference.reshape(1,-1),observed)

    def test_padded_alex_pool_retains_include_pad_denominator(self):
        import numpy as np
        from pt_agent.experiments.heir_cnn_math_check import evaluate_graph
        graph=pool_fixture();x=np.ones((1,2,5,7),dtype=np.float64);x[:,1]*=-2
        observed,_=evaluate_graph(graph,[],x,common_model=graph.name);result=observed.reshape(1,2,3,4)
        self.assertEqual(result[0,0,0,0],4/9)
        self.assertEqual(result[0,0,0,1],6/9)
        self.assertEqual(result[0,0,1,1],1.)
        self.assertEqual(result[0,0,-1,-1],4/9)
        self.assert_math(result[:,0]*-2,result[:,1])

    def test_pool_rejects_changed_default_options_and_wrong_output_shape(self):
        from pt_agent.experiments.heir_common_models import avgpool_spec
        graph=pool_fixture();attrs=graph.nodes[0].attrs;shape=graph.tensors['x'].shape;output=graph.tensors['pool'].shape
        for key,value in [('ceil_mode',True),('count_include_pad',False),('divisor_override',5),('padding',[2,2]),('stride',[0,2])]:
            with self.subTest(key=key),self.assertRaises(ValueError):avgpool_spec(shape,dict(attrs,**{key:value}),output)
        with self.assertRaises(ValueError):avgpool_spec(shape,attrs,(1,2,4,4))

    def test_export_keeps_derived_pool_weights_separate_and_legacy_explicit(self):
        from pt_agent.experiments.heir_cnn_export import export_graph
        from pt_agent.experiments.heir_cnn_math_check import check_constants
        from pt_agent.experiments.heir_common_models import checked_export_contract,graph_contract
        graph=pool_fixture();text,metadata=export_fixture(graph)
        self.assertIn('@alexnet_silu_avgpool(',text)
        self.assertIn('linalg.conv_2d_nchw_fchw',text)
        self.assertIn('[[0], [1, 2, 3]]',text)
        rows=check_constants(graph,text,common_model=graph.name)
        self.assertEqual([r['name'] for r in rows],['derived-exact-local-avgpool:pool'])
        self.assertTrue(all(r['passed'] and r['numerically_equal'] for r in rows))
        checked_export_contract(metadata,graph_contract(graph,graph.name))
        self.assertEqual(metadata['common_model_contract']['original_tensor_constant_count'],0)
        self.assertEqual(metadata['fixed_local_avgpool_lowerings'][0]['divisor'],9)
        with patch('pt_agent.experiments.heir_cnn_export.saved_activations',return_value=(NS(params=[NS(shape=graph.tensors['x'].shape)]),[])):
            with self.assertRaisesRegex(ValueError,'Unsupported original graph operation'):export_graph(graph,'unused')
        changed=copy.deepcopy(metadata);changed['fixed_local_avgpool_lowerings'][0]['divisor']=4
        with self.assertRaises(ValueError):checked_export_contract(changed,graph_contract(graph,graph.name))

    def test_reshape_preserves_channel_row_column_c_order(self):
        import numpy as np
        from pt_agent.experiments.heir_common_models import reshape_spec
        from pt_agent.experiments.heir_cnn_export import Emitter,Tensor
        x=np.empty((1,2,3,5),dtype=np.float64)
        for c in range(2):
            for h in range(3):
                for w in range(5):x[0,c,h,w]=100*c+10*h+w
        graph=NS(name='alexnet_silu_avgpool',inputs=['x'],outputs=['y'],weights={},
            tensors={'x':NS(shape=x.shape),'y':NS(shape=(1,30))},
            nodes=[NS(id='r',op='reshape',inputs=['x'],outputs=['y'],attrs={'shape':[1,30]})])
        from pt_agent.experiments.heir_cnn_math_check import evaluate_graph
        output,_=evaluate_graph(graph,[],x,common_model=graph.name)
        self.assertEqual(output[0].tolist(),[100*c+10*h+w for c in range(2) for h in range(3) for w in range(5)])
        emitter=Emitter();emitter.c_order_reshape(Tensor('%x',x.shape),{'shape':[1,30]},(1,30))
        self.assertIn('[[0], [1, 2, 3]]','\n'.join(emitter.lines))
        for attrs,expected in [({'shape':[1,29]},(1,29)),({'shape':[1,30],'order':'F'},(1,30)),({'shape':[2,15]},(2,15))]:
            with self.assertRaises(ValueError):reshape_spec(x.shape,attrs,expected)

    def test_rank_two_and_four_batchnorm_match_independent_channel_scalars(self):
        import numpy as np
        from pt_agent.experiments.heir_cnn_math_check import evaluate_graph
        for shape in [(1,3),(1,3,2,4)]:
            x=np.arange(np.prod(shape),dtype=np.float64).reshape(shape)/4-2
            weights={'g':np.array([-.5,1.,2.]),'b':np.array([.2,-.3,.7]),'m':np.array([.4,-1.,.8]),'v':np.array([.5,2.,3.])}
            nodes=[NS(id='bn',op='batchnorm',inputs=['x'],outputs=['bn'],attrs={'weight':'g','bias':'b','mean':'m','var':'v','eps':1e-5})]
            tensors={'x':NS(shape=shape),'bn':NS(shape=shape)};output='bn'
            if len(shape)==4:
                nodes.append(NS(id='r',op='reshape',inputs=['bn'],outputs=['out'],attrs={'shape':[1,int(np.prod(shape))]}))
                tensors['out']=NS(shape=(1,int(np.prod(shape))));output='out'
            graph=NS(name='vgg16_silu_avgpool',inputs=['x'],outputs=[output],weights=weights,nodes=nodes,tensors=tensors)
            expected=np.empty_like(x)
            for index in np.ndindex(shape):
                c=index[1];scale=weights['g'][c]/np.sqrt(weights['v'][c]+1e-5)
                expected[index]=x[index]*scale+(weights['b'][c]-scale*weights['m'][c])
            observed,_=evaluate_graph(graph,[],x,common_model=graph.name)
            self.assert_math(expected.reshape(observed.shape),observed)


class CommonSourceRecords(unittest.TestCase):
    def fixture(self):
        from pt_agent.experiments.heir_common_models import graph_contract,expected_constant_names
        graph=pool_fixture();_,metadata=export_fixture(graph);contract=graph_contract(graph,graph.name)
        constants=[dict(name=n,passed=True,numerically_equal=True) for n in expected_constant_names(contract,metadata)]
        refs=[dict(sample_id='train:'+str(i),reference=[[0.]*contract['output_shape'][1]],logit_check={'passed':True},activation_checks=[]) for i in range(3)]
        record={'status':'ok','model':graph.name,'worker':dict(status='ok',common_model=graph.name,
            common_model_contract=contract,activation_pairs=0,reference_forward_completed=3,adapter_forward_completed=3,
            quality_selection=False,constants=constants,probes=refs)}
        return graph,metadata,contract,record

    def test_contract_comes_from_original_metadata_without_weight_reads(self):
        from pt_agent.experiments.heir_common_models import graph_contract,read_graph_contract
        graph=pool_fixture()
        with tempfile.TemporaryDirectory() as name:
            p=Path(name);p.joinpath('graph.json').write_text(json.dumps(dict(name=graph.name,inputs=graph.inputs,outputs=graph.outputs,
                tensors=[[n,list(t.shape),'feature'] for n,t in graph.tensors.items()],
                nodes=[[n.id,n.op,n.inputs,n.outputs,n.attrs,'fixture'] for n in graph.nodes])))
            self.assertFalse((p/'weights.npz').exists())
            self.assertEqual(read_graph_contract(p,graph.name),graph_contract(graph,graph.name))
            with self.assertRaises(ValueError):read_graph_contract(p,'vgg16_silu_avgpool')

    def test_source_records_reject_renamed_model_changed_counts_and_constants(self):
        from pt_agent.experiments.heir_common_models import checked_math_record
        graph,metadata,contract,record=self.fixture()
        self.assertEqual(len(checked_math_record(record,contract,metadata)),3)
        for location,key,value in [('outer','model','resnet20_silu'),('worker','activation_pairs',57),('worker','common_model','vgg16_silu_avgpool')]:
            changed=copy.deepcopy(record);target=changed if location=='outer' else changed['worker'];target[key]=value
            with self.assertRaises(ValueError):checked_math_record(changed,contract,metadata)
        changed=copy.deepcopy(record);changed['worker']['constants'][0]['name']='original-trained-pool'
        with self.assertRaises(ValueError):checked_math_record(changed,contract,metadata)
        changed=copy.deepcopy(record);changed['worker']['probes'][0]['reference'][0][0]=float('nan')
        with self.assertRaises(ValueError):checked_math_record(changed,contract,metadata)

    def test_saved_rule_identity_retains_degree_margin_and_original_program_path(self):
        from pt_agent.experiments.heir_common_models import checked_saved_frontend_identity
        model='vgg16_silu_avgpool';identity='pt_agent.experiments.r13_common_models:'+model
        path=Path('/tmp/synthetic-heir-program.ptdsl')
        saved={'status':'ok','round':'R17a','model':identity,'worker':dict(status='ok',model=identity,
            heldout_acceptance=True,completed_samples=1000,math_probes_passed=True,
            rule_options={'deg':63,'margin':1.5,'slots':16384,'placement':'depth-dp','refresh':'compiler','act_method':'cheb'},
            artifacts={'program':str(path)})}
        checked_saved_frontend_identity(saved,path,model)
        for key,value in [('deg',31),('margin',1.0)]:
            changed=copy.deepcopy(saved);changed['worker']['rule_options'][key]=value
            with self.assertRaises(ValueError):checked_saved_frontend_identity(changed,path,model)
        with self.assertRaises(ValueError):checked_saved_frontend_identity(saved,path,'alexnet_silu_avgpool')
        with self.assertRaises(ValueError):checked_saved_frontend_identity(saved,path.with_name('other.ptdsl'),model)

    def test_complete_common_source_chain_checks_paths_and_original_graph(self):
        from pt_agent.experiments.heir_native_clear_math import checked_sources
        graph,metadata,contract,adapter=self.fixture()
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);graph_dir=root/'graph';export=root/'export';native_dir=root/'native';go_dir=root/'go'
            for p in (graph_dir,export,native_dir,go_dir/'module/heircnn_utils'):p.mkdir(parents=True)
            graph_dir.joinpath('graph.json').write_text(json.dumps(dict(name=graph.name,inputs=graph.inputs,outputs=graph.outputs,
                tensors=[[n,list(t.shape),'feature'] for n,t in graph.tensors.items()],
                nodes=[[n.id,n.op,n.inputs,n.outputs,n.attrs,'fixture'] for n in graph.nodes])))
            program=root/'rule/worker/program.ptdsl';program.parent.mkdir(parents=True);program.write_text('source fixture; never executed')
            identity='pt_agent.experiments.r13_common_models:'+graph.name
            program.parent.parent.joinpath('result.json').write_text(json.dumps(dict(status='ok',round='R17a',model=identity,
                worker=dict(status='ok',model=identity,heldout_acceptance=True,completed_samples=1000,math_probes_passed=True,
                    rule_options={'deg':63,'margin':1.5,'slots':16384,'placement':'depth-dp','refresh':'compiler','act_method':'cheb'},
                    artifacts={'program':str(program)}))))
            mlir=export/'model.mlir';mlir.write_text('source fixture; never compiled')
            metadata['sources']={'graph':str(graph_dir),'program':str(program)}
            export.joinpath('export.json').write_text(json.dumps(metadata))
            native_dir.joinpath('model-lattigo.mlir').write_text('native metadata fixture')
            native={'status':'ok','model_compiled':True,'completed':{'compile':1},'backend_executed':False,
                'model':graph.name,'common_model':graph.name,'common_model_contract':contract,'source':str(mlir)}
            native_path=native_dir/'result.json';native_path.write_text(json.dumps(native))
            go_path=go_dir/'result.json';go_path.write_text(json.dumps(dict(status='ok',model=graph.name,
                library_build_passed=True,input_mlir=str(native_dir/'model-lattigo.mlir'),source_compiler_result=str(native_path))))
            go_dir.joinpath('module/heircnn_utils/heircnn_utils.go').write_text('source metadata fixture')
            probes=root/'saved-probes.npz';probes.touch()
            probe_source=root/'probe-source.json';probe_source.write_text(json.dumps(dict(split='train',indices=[0,1,2],sample_ids=['train:0','train:1','train:2'])))
            go=root/'existing-go';go.touch()
            adapter['worker']['source']=dict(graph=str(graph_dir),program=str(program),mlir=str(mlir),probes=str(probes),probe_source=str(probe_source))
            adapter_path=root/'adapter.json';adapter_path.write_text(json.dumps(adapter))
            args=NS(native_result=native_path,go_result=go_path,adapter_result=adapter_path,probes=probes,probe_source=probe_source,go=go,common_model=graph.name)
            with patch('pt_agent.experiments.heir_cnn_export.saved_activations',return_value=(None,[])):
                ir,_,references,_=checked_sources(args)
                self.assertEqual(ir,native_dir/'model-lattigo.mlir')
                self.assertEqual(len(references),3)
                adapter['worker']['source']['mlir']=str(root/'other-model.mlir')
                adapter_path.write_text(json.dumps(adapter))
                with self.assertRaises(ValueError):checked_sources(args)

    def test_clear_template_uses_selected_entry_and_source_shapes_with_legacy_default(self):
        from pt_agent.experiments.heir_native_clear_codegen import Emitter
        model='alexnet_silu_avgpool';suffixes=['__preprocessing','__preprocessed','__encrypt__arg0','__decrypt__result0']
        fake=NS(program=NS(functions={model+s:None for s in suffixes}))
        code=Emitter.main(fake,model,input_shape=[1,2,5,7],output_shape=[1,24])
        self.assertIn('len(sample.Input)!=70||len(sample.Reference)!=24',code)
        self.assertIn('observed:=make([]float64,24)',code)
        self.assertIn('input:=tensor([]int64{1,2,5,7})',code)
        self.assertIn('f_alexnet_silu_avgpool__preprocessed',code)
        self.assertNotIn('tensor([]int64{1,3,32,32})',code)
        legacy=NS(program=NS(functions={'resnet20_silu'+s:None for s in suffixes}))
        self.assertIn('len(sample.Input)!=3072||len(sample.Reference)!=10',Emitter.main(legacy,'resnet20_silu'))


if __name__=='__main__':unittest.main()
