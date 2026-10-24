"""Original-model CLI, split interfaces and no-label internal validation."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch import nn

from pt_agent import cli
from pt_agent.data.samples import Samples, SampleSplits, StoredSplits, read, save_split
from pt_agent.dsl.parser import parse
from pt_agent.intake.probe import import_torch, model_info
from pt_agent.models.info import ModelInfo, source_kind
from pt_agent.planners.agent_tools import build_context, evaluate, validate
from pt_agent.planners.prompts import build_system_prompt


class PipelineTests(unittest.TestCase):
    def test_source_detection_and_only_original_arguments(self):
        self.assertEqual(source_kind('network.onnx'),'onnx')
        self.assertEqual(source_kind('my_model.py:build'),'torch')
        self.assertEqual(source_kind('package:factory'),'torch')
        self.assertEqual(source_kind('repository/model'),'hf')
        for flag in ('--data','--normalize','--shape','--seq-len','--task'):
            with self.subTest(flag=flag),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as caught:
                cli.main(['--model','example:factory','--inputs','.',flag,'unused'])
            self.assertEqual(caught.exception.code,2)

    def test_original_tensors_and_split_exchange(self):
        with TemporaryDirectory() as tmp:
            directory=Path(tmp)
            x=np.arange(223*4,dtype=np.float64).reshape(223,4)/100
            np.savez(directory/'samples.npz',features=x)
            raw=read(directory/'samples.npz')
            module=nn.Linear(4,2).double().eval()
            imported=import_torch(module,raw,name='original',source='user:factory')
            info=imported.info
            self.assertEqual(info.graph.inputs,['features'])
            info.save(directory/'model')
            recovered=ModelInfo.load(directory/'model')
            self.assertEqual(recovered.inputs,info.inputs)
            self.assertEqual(recovered.source,info.source)
            for split in ('calibration','heldout'):
                arrays,labels=imported.samples.get(split)
                np.testing.assert_array_equal(arrays['features'],x[info.split[split]])
                self.assertIsNone(labels)
                save_split(imported.samples,split,directory/'prepared')
                stored=StoredSplits(directory/'prepared',recovered)
                np.testing.assert_array_equal(stored.get(split)[0]['features'],arrays['features'])
            self.assertFalse((directory/'prepared/confirm.npz').exists())
            self.assertFalse((directory/'prepared/report.npz').exists())

    def test_unlabelled_metrics_and_prompt_keep_original_model(self):
        # Fixed finite input fixtures test interface/metric semantics; no random
        # input is supplied in place of a real calibration/acceptance dataset.
        x=np.tile(np.array([[-1.,2.],[2.,-1.]],dtype=np.float64),(112,1))[:223]
        raw=Samples('tensor',[{'input':value[None]} for value in x],[str(i) for i in range(len(x))])
        module=nn.Linear(2,2,bias=False).double().eval()
        with torch.no_grad():
            module.weight.copy_(torch.eye(2,dtype=torch.float64))
        imported=import_torch(module,raw,name='agreement',source='user:linear')
        ctx=build_context(imported.info,{},samples=imported.samples)
        source='@fhe.program(slots=16384)\ndef agreement(x: Cipher[1,2]):\n    y = pack(x, layout="flat")\n    z = linear(y, w=W["weight"], algo="bsgs")\n    return z\n'
        result=validate(parse(source),ctx,boot_nodes=[])
        self.assertEqual(result['agreement_pct'],100.)
        self.assertNotIn('top1',result)
        self.assertNotIn('acc_drop_pp',result)
        self.assertTrue(result['feasible'])
        text=build_system_prompt(ctx)
        self.assertIn('99.5% (agreement_pct)',text)
        self.assertNotIn('0.5 percentage points below',text)
        feedback=evaluate(source,ctx)
        self.assertEqual(feedback['validation']['agreement_pct'],100.)
        for task, phrase in (('causal_lm','99% (agreement_pct)'),('regression','1e-2 (relative_error)')):
            with self.subTest(task=task):
                imported.info.task=task
                description=build_system_prompt(ctx)
                self.assertIn(phrase,description)
                self.assertNotIn('0.5 percentage points below',description)


if __name__ == '__main__':
    unittest.main()
