"""Small synthetic CNN signature through the real fixed Rule63/Count/Emit/math."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.experiments import cold_rule63_worker as w
from pt_agent.experiments.r10_paper_metrics import StageRecorder,write_json


class ColdRuleWorker(unittest.TestCase):
    def test_real_small_rule_count_emit_and_strict_training_math(self):
        from pt_agent.intake.graph import Graph,Node,TensorInfo
        from pt_agent.models.info import ModelInfo
        from pt_agent.dsl.parser import parse
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);generation=root/'work/generation';generation.mkdir(parents=True)
            intake=root/'work/intake';intake.mkdir();emit=root/'work/emit';emit.mkdir()
            model='resnet20_silu';factory='pt_agent.experiments.r13_common_models:'+model
            tensors={n:TensorInfo(n,shape,'features')for n,shape in
                [('x',(1,3,32,32)),('pooled',(1,3072)),('logits',(1,10)),('y',(1,10))]}
            graph=Graph('synthetic_cold_signature',[
                Node('m','flatten',['x'],['pooled'],dict(start_dim=1,end_dim=-1),'synthetic'),
                Node('l','linear',['pooled'],['logits'],dict(weight='w',bias='b'),'synthetic'),
                Node('a','act',['logits'],['y'],dict(fn='silu'),'synthetic')],tensors,
                dict(w=np.arange(30720,dtype=np.float64).reshape(10,3072)/307200000,b=np.arange(10,dtype=np.float64)/100),
                ['x'],['y'])
            info=ModelInfo(dict(factory=factory),graph,{},[dict(name='x',kind='features',dtype='float64',
                shape=[1,3,32,32])],None,'classification','top1',{},dict(calibration='synthetic fixture'))
            info.save(intake/'model_info')
            calibration=root/'calibration.json';write_json(calibration,dict(logits=dict(min=-.5,max=.5)))
            probes=np.stack([np.full((3,32,32),value,dtype=np.float64)for value in(.1,.2,.3)])
            np.savez(intake/'training_probes.npz',input=probes)
            write_json(intake/'phase_result.json',dict(artifacts=dict(training_probes=str(intake/'training_probes.npz'))))
            args=SimpleNamespace(model=model,model_info=intake/'model_info/model_info.json',calibration=calibration,
                program=generation/'program.ptdsl',case_root=root,emit_dir=emit,frontend=root/'unused.py',case_gib=64)
            rec=StageRecorder(generation/'events.jsonl');result={}
            w.generate(args,rec,result)
            self.assertTrue(all(row['passed']for row in result['training_math']['checks']))
            self.assertEqual(result['rule_options']['margin'],1.5)
            self.assertEqual(result['raw_Count']['stats']['counts']['boot'],0)
            program=parse(args.program.read_text());acts=[op for op in program.ops if op.kind=='act']
            self.assertEqual(len(acts),1);self.assertEqual(acts[0].attrs['deg'],63)
            self.assertEqual(acts[0].attrs['domain'],(-.75,.75))
            write_json(generation/'phase_result.json',result)
            emitted={}
            with patch.object(w.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)):
                w.emit(args,StageRecorder(emit/'events.jsonl'),emitted)
            self.assertEqual(emitted['emit_stats']['counts'],result['raw_Count']['stats']['counts'])
            self.assertTrue((emit/'candidate_trace.py').is_file());self.assertTrue((emit/'constants.npz').is_file())
            self.assertFalse((emit/'candidate_trace.mlir').exists())

    def test_inner_phase_exception_persists_error_without_registry_or_retry(self):
        from pt_agent.experiments import r30_common_model_e2e as r30
        from pt_agent.experiments import r29_local_compile_matrix as r29
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);recipe=root/'recipe.json';write_json(recipe,[])
            args=SimpleNamespace(work_dir=root/'phase',model='resnet20_silu',worker_phase='intake',metadata_template=recipe)
            with patch.object(r29,'native_metadata_commands',return_value=nullcontext()),\
                    patch.object(r30,'_intake',side_effect=RuntimeError('Synthetic fresh source failure'))as intake:
                self.assertEqual(w.run(args),1)
            value=w.read(args.work_dir/'phase_result.json')
            self.assertEqual(value['status'],'error');self.assertIn('Synthetic fresh source failure',value['error'])
            self.assertEqual(intake.call_count,1);self.assertEqual(value['api_calls'],0)
            self.assertFalse((root/'registry.jsonl').exists())


if __name__=='__main__':unittest.main(verbosity=2)
