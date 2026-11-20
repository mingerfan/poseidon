"""Small owned-array and mapped-weight reference fixtures."""

import gc
import os
from pathlib import Path
import tempfile
import unittest
import weakref


class MappedReferenceCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name]='1'
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from pt_agent.intake.graph import Graph,Node,TensorInfo
        from pt_agent.models.info import ModelInfo
        from pt_agent.models.mapped_info import load_mapped_info
        from pt_agent.intake.mapped_reference import MappedReference
        from pt_agent.intake import graph_exec
        cls.np,cls.Graph,cls.Node,cls.Tensor=np,Graph,Node,TensorInfo
        cls.Info,cls.load,cls.Reference=ModelInfo,staticmethod(load_mapped_info),MappedReference
        cls.original=graph_exec

    def fixture(self):
        weights={'scale':self.np.asarray([.9,1.1,1.2]),
                 'weight':self.np.asarray([[.1,-.2,.3],[.4,.2,-.1]])}
        tensors={name:self.Tensor(name,shape,'input' if name=='input' else 'act')
                 for name,shape in [('input',(1,3)),('norm',(1,3)),('output',(1,2))]}
        graph=self.Graph('fixture',[self.Node('n0','rmsnorm',['input'],['norm'],{'weight':'scale','eps':1e-6},'synthetic'),
            self.Node('n1','linear',['norm'],['output'],{'weight':'weight'},'synthetic')],tensors,weights,['input'],['output'])
        info=self.Info(source={'type':'torch','path':'synthetic:original'},graph=graph,input_handling={},
            inputs=[{'name':'input','shape':[1,3],'kind':'features','dtype':'float64'}],sequence=None,
            task='classification',metric='classification_top1',threshold={'acc_drop_pp':.5},
            split={'calibration':[0],'heldout':[1],'confirm':[2],'report':[3]},assumptions=['original metadata'])
        return info,{'input':self.np.asarray([[.2,-.3,.4]])}

    def test_mapped_info_keeps_all_original_metadata_and_readonly_weight_values(self):
        info,inputs=self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            info.save(temporary)
            mapped=self.load(temporary)
            for name in ('source','input_handling','inputs','sequence','task','metric','threshold','split','assumptions'):
                self.assertEqual(getattr(mapped,name),getattr(info,name))
            self.assertEqual(mapped.mapped_weights['members'],2)
            self.assertEqual(mapped.mapped_weights['full_tensor_copies'],0)
            for name,value in mapped.graph.weights.items():
                self.assertIsInstance(value,self.np.memmap)
                self.assertFalse(value.flags.writeable)
                self.np.testing.assert_array_equal(value,info.graph.weights[name])
                with self.assertRaises(ValueError):value.flags.writeable=True

    def test_external_mutable_weights_and_metadata_cannot_change_private_reference(self):
        info,inputs=self.fixture()
        expected=self.original.forward(info.graph,inputs)['output']
        reference=self.Reference(info.graph)
        info.graph.weights['weight'][:]=100.
        info.graph.nodes[0].attrs['eps']=9.
        observed=reference.forward(inputs)['output']
        self.assertTrue(self.np.all(self.np.abs(expected-observed)<=1e-12+1e-12*self.np.abs(expected)))
        self.assertEqual(reference.observation['private_weight_copies'],2)
        self.assertTrue(reference.observation['all_numpy_weights_readonly'])
        self.assertIs(reference._forward.__code__,self.original.forward.__code__)
        calibration=reference.forward(inputs,keep=True)
        self.assertNotIn('weight',calibration)
        self.assertNotIn('scale',calibration)
        self.assertIn('norm',calibration)

    def test_mapped_weights_share_views_and_outlive_original_graph_until_executor_release(self):
        info,inputs=self.fixture()
        expected=self.original.forward(info.graph,inputs)['output']
        with tempfile.TemporaryDirectory() as temporary:
            info.save(temporary);mapped=self.load(temporary)
            reference=self.Reference(mapped.graph)
            watched=weakref.ref(mapped.graph.weights['weight'])
            del mapped
            gc.collect();self.assertIsNotNone(watched())
            observed=reference.forward(inputs)['output']
            self.assertTrue(self.np.all(self.np.abs(expected-observed)<=1e-12+1e-12*self.np.abs(expected)))
            self.assertEqual(reference.observation['mapped_weight_views'],2)
            self.assertEqual(reference.observation['private_weight_copies'],0)
            self.assertEqual(reference.observation['interpreter_weight_view_calls'],2)
            del reference;gc.collect()
            self.assertIsNone(watched())
            self.assertTrue(self.np.isfinite(observed).all())


if __name__=='__main__':unittest.main()
