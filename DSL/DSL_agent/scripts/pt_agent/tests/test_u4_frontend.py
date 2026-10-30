"""Generic attention normalization and the retained BERT semantic graph."""

import unittest
import numpy as np
import torch
from torch import nn

from pt_agent.intake import import_module
from pt_agent.intake.transformer import normalize
from pt_agent.intake.graph_exec import forward
from pt_agent.intake.graph import Graph
from pt_agent.intake.coverage import report,check
from pt_agent.paths import pt_result
from pt_agent.tests._artifacts import require


class NormalizationTests(unittest.TestCase):
    def test_mha_views_and_mask(self):
        class Attention(nn.Module):
            def forward(self,x,m):
                q=x.reshape(1,3,2,4).permute(0,2,1,3)
                p=torch.softmax((q@q.transpose(-1,-2))*.5+(1-m[:,None,None,:])*(-10000),-1)
                return (p@q).permute(0,2,1,3).reshape(1,3,8)
        x=torch.arange(24,dtype=torch.float64).reshape(1,3,8)/100
        m=torch.tensor([[1,1,0]],dtype=torch.float64)
        raw=import_module(Attention(),(x,m),'attention',frontend='aten')
        graph=normalize(raw)
        self.assertTrue(check(graph)['ok'])
        self.assertEqual(report(graph)['lowering_ratio'],1)
        self.assertFalse(set(('reshape','permute','broadcast')) & {n.op for n in graph.nodes})
        actual=forward(graph,dict(zip(graph.inputs,(x.numpy(),m.numpy()))))[graph.outputs[0]]
        np.testing.assert_allclose(actual,Attention()(x,m).numpy(),atol=1e-9,rtol=1e-9)
        soft=next(n for n in graph.nodes if n.op=='softmax')
        self.assertEqual(soft.attrs['keep'],'all')
        self.assertEqual(soft.attrs['mask'],graph.inputs[1])

    def test_illegal_mask_stops_at_check(self):
        class Bad(nn.Module):
            def forward(self,x,m):
                q=x.reshape(1,3,2,4).permute(0,2,1,3)
                return torch.softmax(q@q.transpose(-1,-2)+(1-m[:,None,None,:])*(-17),-1)
        x=torch.arange(24,dtype=torch.float64).reshape(1,3,8)/100
        m=torch.tensor([[1,1,0]],dtype=torch.float64)
        graph=normalize(import_module(Bad(),(x,m),'bad',frontend='aten'),strict=False)
        result=check(graph)
        self.assertFalse(result['ok'])
        self.assertIn('掩码',result['diags'][0]['reason'])
        self.assertIsNotNone(result['diags'][0]['op_id'])
        self.assertGreater(result['diags'][0]['lineno'],0)


BERT=pt_result('u1-cli-20261024T104532576122/bert_tiny')


@require(BERT/'graph/graph.json',BERT/'samples/calibration.npz')
class RetainedBertTests(unittest.TestCase):
    def test_normalized_bert_reference(self):
        raw=Graph.load(BERT/'graph')
        graph=normalize(raw)
        self.assertEqual(report(graph)['lowering_ratio'],1)
        self.assertTrue(check(graph)['ok'])
        with np.load(BERT/'samples/calibration.npz',allow_pickle=False) as f:
            arrays={k:f[k][:2] for k in raw.inputs}
        expected=np.concatenate([forward(raw,{k:v[i:i+1] for k,v in arrays.items()})[raw.outputs[0]]
                                 for i in range(2)])
        actual=forward(graph,arrays)[graph.outputs[0]]
        np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-9)
        self.assertEqual(sum(n.op=='layernorm' for n in graph.nodes),5)
        self.assertEqual(sum(n.op=='softmax' for n in graph.nodes),2)


if __name__=='__main__':unittest.main()
