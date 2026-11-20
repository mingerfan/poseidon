"""Only shape metadata and fake placement; no original model or real Count."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.dsl.parser import parse
from pt_agent.dsl.ir import canonical
from pt_agent.experiments import a7_qwen_layer0_sweep as m
from pt_agent.experiments import qwen_layered_structure as layered


def fixture():
    source='''def qwen_layer(x,am,w,n1,n2):
 a=rmsnorm(x,w=W[n1],eps=1e-6,rsqrt_method="newton",rsqrt_domain=(0.01,1.0),rsqrt_deg0=63,rsqrt_iters=4,rsqrt_fit="relative")
 q=matmul(a,a,transpose_b=True,heads=14,kv_heads=14)
 p=softmax(q,axis=-1,keep="causal",mask=am,shift=1.,inv_scale=2.,steps=2,exp_method="cheb",exp_domain=(-1.,1.),exp_deg=31,inv_method="newton",inv_domain=(0.1,2.),inv_deg0=63,inv_iters=4,step_inv_method="newton",step_inv_domain=(0.01,2.),step_inv_deg0=63,step_inv_iters=4)
 v=matmul(p,a,transpose_b=False,heads=14,kv_heads=14)
 n=rmsnorm(v,w=W[n2],eps=1e-6,rsqrt_method="newton",rsqrt_domain=(0.02,2.0),rsqrt_deg0=63,rsqrt_iters=4,rsqrt_fit="relative")
 s=act(n,fn="silu",method="cheb",deg=63,domain=(-2.,3.))
 y=linear(s,w=W[w])
 out=add(y,x)
 return out
@fhe.program(slots=16384,refresh="agent",placement="program-plan",l_max=5,boot_out_level=5)
def qwen2(inputs_embeds: Cipher[1,32,896],attention_mask: Mask[1,32],sel: Mask[1,32]):
 x=pack(inputs_embeds,layout="token")
 am=pack(attention_mask,layout="flat")
 sm=pack(sel,layout="flat")
'''
    current='x';shapes={}
    for i in range(24):
        prefix=f'original.model.layers.{i}.'
        source+=f' h{i}=qwen_layer({current},am,"{prefix}w","{prefix}norm1","{prefix}norm2")\n'
        shapes.update({prefix+'w':(896,896),prefix+'norm1':(896,),prefix+'norm2':(896,)})
        current='h'+str(i)
    source+=(f' n=rmsnorm({current},w=W["original.model.norm.weight"],eps=1e-6,rsqrt_method="newton",rsqrt_domain=(0.01,2.),rsqrt_deg0=7,rsqrt_iters=2)\n'
        ' selected=select(n,sm)\n logits=linear(selected,w=W["original.model.embed_tokens.weight"])\n return logits\n')
    shapes.update({'original.model.norm.weight':(896,),'original.model.embed_tokens.weight':(151936,896)})
    return parse(source),shapes


def fake_measurement(callback):
    source=layered.source_binding(callback.segment)
    statements=[dict(id=r['id'],boots=50 if i==3 else 0) for i,r in enumerate(source['original_ops'])]
    return dict(role='layer',index=0,boundary_boots=32,output_shape=[1,32,896],output_levels=[5]*32,
        internal=dict(boots_pred=50,max_depth=5,latency_pred_us=1000000),
        complete_count=dict(counts={'boot':82},ops_total=100,rot_steps=[],distinct_consts=2),
        source=source,statements=statements,constants=('a','b'),plan_artifact='fixture/refresh-plan.json',boot_us=100.)


class Tests(unittest.TestCase):
    def setUp(self):
        self.program,self.shapes=fixture();self.partition=layered.split_program(self.program,self.shapes)

    def test_seven_typed_variants_domains_and_other_layers_unchanged(self):
        original=deepcopy(self.program);ids=set(self.partition.layers[0].original_op_ids[3:])
        for name in m.VARIANTS:
            changed,rows=m.variant_program(self.program,self.partition,name)
            part=layered.split_program(changed,self.shapes)
            self.assertEqual(part.layers[0].program.params[0].shape,(1,32,896))
            self.assertEqual([op for op in changed.ops if op.id not in ids],[op for op in original.ops if op.id not in ids])
            for before,after in zip(original.ops,changed.ops):
                for key,value in before.attrs.items():
                    if 'domain' in key:self.assertEqual(value,after.attrs[key])
            self.assertEqual(len(rows),0 if name=='baseline' else 2 if name.startswith('rms') else 1)
        self.assertEqual(self.program,original)

    def test_only_requested_fields_change_and_original_plan_header_kept(self):
        for name in m.VARIANTS:
            changed,rows=m.variant_program(self.program,self.partition,name)
            self.assertEqual(changed.header,self.program.header)
            allowed=({'exp_deg'} if name.startswith('softmax') else {'rsqrt_method','rsqrt_deg','rsqrt_deg0','rsqrt_iters'}
                if name.startswith('rms') else {'deg'} if name.startswith('silu') else set())
            self.assertTrue(all(set(row['changes'])<=allowed for row in rows))
        with self.assertRaises(ValueError):m.variant_program(self.program,self.partition,'steps4')

    def test_wrong_original_RMS_recipe_rejected(self):
        first=self.partition.layers[0].original_op_ids[3]
        ops=tuple(replace(op,attrs=dict(op.attrs,rsqrt_iters=3)) if op.id==first else op for op in self.program.ops)
        with self.assertRaises(ValueError):m.variant_program(replace(self.program,ops=ops),self.partition,'rms-remez15')

    def test_scalar_scope_quality_null_boundary_and_attribution(self):
        callback=layered.callbacks(self.partition,{})[0][0];raw=fake_measurement(callback)
        result=m.scalar_result('baseline',raw,[],.2)
        self.assertEqual(result['boots_pred'],82);self.assertEqual(result['internal_boots_pred'],50)
        self.assertEqual(result['boots_pred_by_component']['boundary'],32)
        self.assertEqual(result['boots_pred_by_component']['norm'],50)
        self.assertAlmostEqual(result['latency_pred_s'],1.0032)
        self.assertIsNone(result['feasible']);self.assertIsNone(result['quality']['value'])
        self.assertTrue(result['placement_feasible']);self.assertFalse(result['C2_inherited'])
        self.assertTrue(all(v is None for v in result['depth_by_component'].values()))
        for key,value in [('boundary_boots',31),('output_levels',[4]*32),('output_shape',[1,16,896])]:
            with self.assertRaises(ValueError):m.scalar_result('baseline',dict(raw,**{key:value}),[],0)

    def test_fake_run_calls_fresh_baseline_and_six_variants_once(self):
        with TemporaryDirectory() as t:
            root=Path(t);binding=root/'binding.json';binding.write_text(json.dumps(layered.source_binding(self.partition.layers[0])))
            args=SimpleNamespace(work_dir=root,baseline_binding=binding);calls=[]
            weights={k:SimpleNamespace(shape=v) for k,v in self.shapes.items()}
            def api(name,directory):
                calls.append(name)
                return SimpleNamespace(boot_us=100.,measure=fake_measurement)
            with patch.object(m,'source_gate',return_value={'fixture':'same'}):
                rows=m.run_variants(args,self.program,weights,api=api)
            self.assertEqual(calls,list(m.VARIANTS));self.assertEqual(len(rows),7)
            self.assertEqual([r['variant'] for r in rows],list(m.VARIANTS))
            self.assertEqual(m.read(root/'progress.json')['completed_variants'],list(m.VARIANTS))
            self.assertTrue(all(m.read(root/name/'result.json')['status']=='ok' for name in m.VARIANTS))

    def test_fake_error_preserves_completed_baseline_and_stops_next(self):
        with TemporaryDirectory() as t:
            root=Path(t);binding=root/'binding.json';binding.write_text(json.dumps(layered.source_binding(self.partition.layers[0])))
            args=SimpleNamespace(work_dir=root,baseline_binding=binding);calls=[]
            def api(name,directory):
                calls.append(name)
                def measure(callback):
                    if name!='baseline':raise RuntimeError('fixture Count error')
                    return fake_measurement(callback)
                return SimpleNamespace(boot_us=100.,measure=measure)
            with patch.object(m,'source_gate',return_value={'fixture':'same'}):
                with self.assertRaisesRegex(RuntimeError,'fixture Count error'):
                    m.run_variants(args,self.program,{k:SimpleNamespace(shape=v) for k,v in self.shapes.items()},api=api)
            self.assertEqual(calls,['baseline','softmax-exp7'])
            self.assertTrue((root/'baseline/result.json').is_file());self.assertFalse((root/'softmax-exp15').exists())
            self.assertEqual(m.read(root/'progress.json')['active_variant'],'softmax-exp7')

    def test_publication_deduplicates_exact_identity_and_rejects_conflict(self):
        with TemporaryDirectory() as t:
            p=Path(t)/'sweep.jsonl';row=dict(run_id='fixture',variant='baseline',feasible=None)
            self.assertTrue(m.append_sweep(p,row));self.assertFalse(m.append_sweep(p,row))
            with self.assertRaises(ValueError):m.append_sweep(p,dict(row,feasible=True))
            self.assertEqual(len(p.read_text().splitlines()),1)

    def test_source_metadata_gate_keeps_exact_original_paths_profile_and_layer(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            info=root/'qwen-normalization-rebased-20261115/work/import/model_info';info.mkdir(parents=True)
            source=root/'qwen-guarded-reply13-rss16-recovery-20261116/session/sources/iteration-13.ptdsl'
            source.parent.mkdir(parents=True);source.write_text('trusted fixture source not executed')
            binding=root/'binding.json';binding.write_text(json.dumps({'format':'qwen-layer-source-binding-v1','role':'layer','index':0}))
            record=dict(source=dict(config=dict(model_type='qwen2',num_hidden_layers=24,hidden_size=896,vocab_size=151936)),
                inputs=[dict(name=name,shape=shape) for name,shape in [('inputs_embeds',[1,32,896]),('attention_mask',[1,32]),('sel',[1,32])]])
            (info/'model_info.json').write_text(json.dumps(record))
            args=SimpleNamespace(model_info=info,source_program=source,profile=root/'profiles/qwen48/compiler-config.json',baseline_binding=binding)
            with patch.object(m.paths,'RESULTS_DIR',root),patch.object(m.paths,'PROFILES_DIR',root/'profiles'):
                checked=m.source_gate(args);self.assertFalse(checked['C2_inherited']);self.assertEqual(checked['prefill_tokens'],32)
                binding.write_text(json.dumps({'format':'qwen-layer-source-binding-v1','role':'head','index':None}))
                with self.assertRaises(ValueError):m.source_gate(args)
                args.source_program=root/'foreign.ptdsl'
                with self.assertRaises(ValueError):m.source_gate(args)

    def test_outer_fake_success_and_failure_close_registry_and_keep_partial(self):
        for failed in (False,True):
            with TemporaryDirectory() as t:
                root=Path(t);args=m.parser().parse_args(['--out',str(root/'case'),
                    '--sweep-output',str(root/'paper/fig2_sweep.jsonl'),'--seconds','60'])
                source={'fixture':'original'};commands=[]
                def process(command,**kwargs):
                    commands.append((command,kwargs));work=args.out/'work'
                    selected=m.VARIANTS[:1] if failed else m.VARIANTS
                    for name in selected:
                        (work/name).mkdir();m.write_json(work/name/'result.json',
                            dict(status='ok',source=source,variant=name,boots_pred=82,quality=None))
                    m.write_json(work/'worker-result.json',dict(status='error' if failed else 'ok'))
                    return dict(exit_code=1 if failed else 0,timed_out=False,signal=None,remaining_pids=[],signals_sent=[])
                watcher=SimpleNamespace(start=lambda:None,is_alive=lambda:False)
                with patch.object(m.paths,'RESULTS_DIR',root),patch.object(m,'source_gate',return_value=source),\
                        patch.object(m.threading,'Thread',return_value=watcher),patch.object(m,'run_process',side_effect=process),\
                        patch('shutil.disk_usage',return_value=SimpleNamespace(free=100*m.GIB)),\
                        patch('pt_agent.registry.git_rev',return_value='fixture'):
                    result=m.execute(args)
                self.assertEqual(result['status'],'error' if failed else 'ok')
                self.assertEqual(result['completed'],0 if failed else 1)
                self.assertEqual(result['attempted'],1);self.assertTrue(result['monitor_stopped'])
                self.assertEqual(len(result['variants']),1 if failed else 7)
                self.assertEqual(commands[0][1]['mem_gib'],24)
                self.assertEqual(commands[0][1]['fsize_gib'],1)
                self.assertLessEqual(commands[0][1]['timeout'],50)
                self.assertIn('--python',commands[0][0]);self.assertIn(str(args.intake_python),commands[0][0])
                rows=[json.loads(line) for line in (root/'registry.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),1);self.assertEqual(set(rows[0]),
                    {'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
                self.assertEqual(m.read(args.out/'completion.json')['status'],result['status'])
                runs=[json.loads(line) for line in (root/'paper/runs.jsonl').read_text().splitlines()]
                self.assertEqual(runs[0]['status'],'driver_error' if failed else 'ok')
                self.assertIsNone(runs[0]['boots_compiled']);self.assertEqual(runs[0]['api_calls'],0)


if __name__=='__main__':unittest.main()
