"""A1 own bounded fixtures; no original model/sample/compiler execution."""
import ast
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import a1_bert_fixed_current as m


RULE='''@fhe.program(slots=16384,refresh="compiler",placement="depth-dp")
def bert(x: Cipher[1,40,128],m: Mask[1,40]):
    a=pack(x,layout="token")
    y=act(a,fn="gelu",method="cheb",deg=127,domain=(-4.0,7.0))
    z=softmax(y,exp_method="remez",exp_deg=7,inv_method="newton",inv_deg0=7,inv_iters=3)
    n=layernorm(z,rsqrt_method="remez",rsqrt_deg=31,rsqrt_domain=(0.1,2.0))
    t=act(n,fn="tanh",method="cheb",deg=127,domain=(-3.0,3.0))
    return t
'''


class Tests(unittest.TestCase):
    def test_only_activation_degree_changes(self):
        changed,rows=m.rule63_source(RULE)
        expected=ast.parse(RULE)
        for node in ast.walk(expected):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='act':
                for key in node.keywords:
                    if key.arg=='deg':key.value=ast.Constant(value=63)
        self.assertEqual(ast.dump(ast.parse(changed)),ast.dump(expected))
        self.assertEqual([r['degree'] for r in rows],[63,63])
        self.assertEqual([r['original_degree'] for r in rows],[127,127])
        self.assertEqual(rows[0]['domain'],(-4.,7.))

    def test_wrong_activation_recipe_rejected(self):
        for text in (RULE.replace('method="cheb"','method="remez"'),RULE.replace('fn="gelu"','fn="silu"')):
            with self.assertRaises(ValueError):m.rule63_source(text)
        with self.assertRaises(ValueError):m.rule63_source('def empty(x):\n return x\n')

    def test_recipe_real_WeightRef_scalar_only_JSON(self):
        from pt_agent.dsl.ir import WeightRef,ValueRef
        op=SimpleNamespace(id=5,lineno=6,kind='layernorm',attrs=dict(w=WeightRef('original.bn.weight'),
            b=WeightRef('original.bn.bias'),rsqrt_method='newton',rsqrt_deg0=7,rsqrt_iters=3,
            rsqrt_domain=(.1,2.),eps=1e-12,mask=ValueRef('mask')))
        row=m.recipe_metadata(SimpleNamespace(ops=[op]))[0]
        encoded=json.loads(json.dumps(row,allow_nan=False))
        self.assertEqual(encoded['attrs']['rsqrt_domain'],[.1,2.])
        self.assertNotIn('w',encoded['attrs'])
        self.assertEqual(encoded['references']['w'],dict(kind='weight',name='original.bn.weight'))
        self.assertEqual(encoded['references']['mask'],dict(kind='value',name='mask'))
        self.assertEqual(encoded['attrs']['rsqrt_iters'],3)

    def args(self,root):
        p=m.parser();args=p.parse_args(['--out',str(root/'out'),'--comparison-case',str(root/'comparison')])
        args.profile=root/'profiles/qwen48/compiler-config.json'
        args.frontend=root/'dacapo/python/hecate/hecate/expr.py'
        args.metadata_template=root/'template.json'
        args.work_dir=args.out/'work';args.program=args.work_dir/'rule63.ptdsl';args.model_info=root/'model/model_info.json'
        args.system='dacapo-depth-dp';args.debug_snapshots=False
        return args

    def fixture(self,root):
        args=self.args(root);args.comparison_case.mkdir()
        args.metadata_template.write_text('[]')
        args.model_info.parent.mkdir();(args.model_info.parent/'calibration.json').write_text('{}')
        info=dict(source=dict(local_path=str(root/'original-model'),config=dict(model_type='bert',
            num_hidden_layers=2,hidden_size=128,num_labels=2)),task='classification',inputs=[],sequence=dict(length=40))
        args.model_info.write_text(json.dumps(info));plan=root/'existing-plan.json';plan.write_text('{}')
        current=dict(status='ok',compiled_bootstraps=9180,runtime_plan=str(plan),source=dict(model_info=str(args.model_info)),
            phases=dict(compile=dict(status='ok',runtime_plan=str(plan))))
        (args.comparison_case/'result.json').write_text(json.dumps(current))
        (args.comparison_case/'completion.json').write_text(json.dumps(dict(status='ok',
            required_artifacts_persisted=True,registry_persisted=True)))
        (args.comparison_case/'work').mkdir()
        (args.comparison_case/'work/compile-argv.json').write_text(json.dumps(['/current/hecate-opt','input',
            '--pars','--waterline=43','--ckks-config='+str(args.profile),'--runtime-plan-context-id=qwen-bootstrap48']))
        return args

    def test_source_same_original_model_profile_and_completion(self):
        with TemporaryDirectory() as t:
            root=Path(t);args=self.fixture(root)
            with patch.object(m.paths,'PROFILES_DIR',root/'profiles'),patch.object(m.paths,'DACAPO_ROOT',root/'dacapo'),\
                    patch.object(m.original,'compiler_argv',return_value=['validated']) as validate:
                source=m.source_gate(args)
                self.assertEqual(source['hecate_opt'],'/current/hecate-opt')
                self.assertFalse(source['quality_evaluated']);self.assertIsNone(source['accuracy'])
                self.assertEqual(validate.call_args.args[1],'dacapo-depth-dp')
                completion=args.comparison_case/'completion.json';completion.write_text('{"status":"error"}')
                with self.assertRaises(ValueError):m.source_gate(args)

    def test_source_foreign_model_and_profile_rejected(self):
        with TemporaryDirectory() as t:
            root=Path(t);args=self.fixture(root)
            with patch.object(m.paths,'PROFILES_DIR',root/'profiles'),patch.object(m.paths,'DACAPO_ROOT',root/'dacapo'),\
                    patch.object(m.original,'compiler_argv',return_value=[]):
                data=json.loads(args.model_info.read_text());data['source']['config']['hidden_size']=256
                args.model_info.write_text(json.dumps(data))
                with self.assertRaises(ValueError):m.source_gate(args)
                data['source']['config']['hidden_size']=128;args.model_info.write_text(json.dumps(data))
                args.profile=root/'foreign.json'
                with self.assertRaises(ValueError):m.source_gate(args)

    def test_commands_explicit_module_intake_phase_limits(self):
        with TemporaryDirectory() as t:
            args=self.args(Path(t))
            for phase in m.PHASES:
                command,seconds=m.phase_command(args,phase)
                self.assertIn(m.MODULE,command);self.assertIn(str(args.intake_python),command)
                self.assertEqual(command[command.index('--memory-gib')+1],'24')
                self.assertEqual(command[command.index('--case-gib')+1],'32')
                self.assertEqual(command[command.index('--worker-phase')+1],phase)
                self.assertEqual(command[command.index('--program')+1],str(args.program))
                self.assertEqual(seconds,7440 if phase=='compile' else 2040)

    def test_real_trusted_rule_parser_typecheck_and_small_Count(self):
        from pt_agent.intake.graph import Graph,Node,TensorInfo
        from pt_agent.models.info import ModelInfo
        tensors={key:TensorInfo(key,shape,'act') for key,shape in
                 [('x',(1,1,2)),('a',(1,1,2)),('y',(1,2))]}
        graph=Graph('tiny_rule',[Node('n0','act',['x'],['a'],dict(fn='tanh'),''),
            Node('n1','flatten',['a'],['y'],dict(start_dim=1,end_dim=-1),'')],tensors,{},['x'],['y'])
        graph.intake_stats=dict(transformer_normalized=True)
        info=SimpleNamespace(graph=graph,inputs=[dict(name='x',shape=[1,1,2],kind='tokens')])
        with TemporaryDirectory() as t:
            root=Path(t);args=self.args(root);args.work_dir.mkdir(parents=True)
            calibration=root/'calibration.json';calibration.write_text('{"x":{"min":-0.5,"max":0.5}}')
            args.metadata_template.write_text('[]')
            source=dict(model_info=str(args.model_info),calibration=str(calibration))
            with patch.object(m,'source_gate',return_value=source),patch.object(ModelInfo,'load',return_value=info),\
                    patch.object(m.original,'native_metadata_commands',return_value=nullcontext()):
                result=m.prepare(args)
            self.assertEqual(result['status'],'ok')
            self.assertEqual(result['recipe'][0]['attrs']['deg'],63)
            self.assertEqual(result['activation_degree_changes'][0]['original_degree'],127)
            self.assertGreater(result['raw_count']['ops_total'],0)
            self.assertEqual(result['raw_count']['counts'].get('boot',0),0)
            self.assertEqual(result['raw_count']['outputs'],1)

    def test_raw_count_and_emit_entire_inventory_comparison(self):
        with TemporaryDirectory() as t:
            root=Path(t);args=self.args(root);args.worker_phase='emit';args.work_dir.mkdir(parents=True)
            source=dict(model_info=str(args.model_info),hecate_opt='/compiler')
            stats=dict(counts={'mul_cc':3},ops_total=3,distinct_consts=2)
            args.program.write_text(RULE)
            (args.work_dir/'prepare-result.json').write_text(json.dumps(dict(status='ok',source=source,
                program_ast=ast.dump(ast.parse(RULE),include_attributes=False),raw_count=stats)))
            def fake(args):
                m.write_json(args.work_dir/'emit-result.json',dict(round=m.ROUND,mode=args.system,model=m.MODEL,
                    status='ok',emit_stats=stats));return 0
            # This closure is installed as the private original worker code, so
            # patch only its module-local names, without reading a real graph.
            with patch('pt_agent.experiments.bert_sst2_worker.initialize_numeric'),patch.object(m,'source_gate',return_value=source),\
                    patch.object(m.original,'worker',fake):
                # The production code clones its actual ordinary function;
                # provide this fixture's required module global explicitly.
                with patch.dict(vars(m.original),m=m):
                    self.assertEqual(m.worker(args),0)
                    (args.work_dir/'emit-result.json').rename(args.work_dir/'accepted-emit-result.json')
                    stats['ops_total']=4
                    self.assertEqual(m.worker(args),1)
            self.assertIn('Count/Emit',m.read(args.work_dir/'emit-result.json')['error'])

    def test_execute_saved_success_registry_and_clock(self):self.execute_fixture(False)
    def test_execute_first_error_preserves_attempts_and_stops(self):self.execute_fixture(True)

    def execute_fixture(self,fail):
        with TemporaryDirectory() as t:
            root=Path(t);args=self.args(root)
            for f in (args.profile,args.metadata_template):f.parent.mkdir(parents=True,exist_ok=True);f.write_text('{}')
            source=dict(model_info=str(args.model_info));(root/'registry.jsonl').write_text('')
            def process(command,**kwargs):
                phase=command[-1]
                row=dict(round=m.ROUND,model=m.MODEL,mode=m.MODE,source=source,phase=phase,
                    status='error' if fail and phase=='emit' else 'ok',error='fixture' if fail else None)
                if phase=='compile':row.update(compiled_bootstraps=12,predicted_latency_us=8,runtime_plan='fixture',
                    process=dict(started_ns=10,ended_ns=2000000010))
                m.write_json(args.work_dir/(phase+'-result.json'),row)
                return dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[])
            watcher=SimpleNamespace(start=lambda:None,is_alive=lambda:False)
            with patch.object(m.paths,'RESULTS_DIR',root),patch.object(m,'source_gate',return_value=source),\
                    patch.object(m,'phase_command',side_effect=lambda a,p:([p],2)),patch.object(m,'run_process',side_effect=process),\
                    patch.object(m.threading,'Thread',return_value=watcher),patch('pt_agent.registry.git_rev',return_value='fixture'):
                result=m.execute(args)
            self.assertEqual(result['status'],'error' if fail else 'ok')
            self.assertEqual(result['compile_accepted'],not fail)
            self.assertEqual(result['attempted']['compile'],0 if fail else 1)
            self.assertEqual(result['compile_s'],None if fail else 2.)
            self.assertEqual(len((root/'registry.jsonl').read_text().splitlines()),1)
            self.assertEqual(set(json.loads((root/'registry.jsonl').read_text())),
                {'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
            self.assertTrue(m.read(args.out/'completion.json')['registry_persisted'])


if __name__=='__main__':unittest.main()
