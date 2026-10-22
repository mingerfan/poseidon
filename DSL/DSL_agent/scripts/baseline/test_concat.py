"""Concat geometry, independent Torch/reference, layout extraction and refusals."""
import copy
import itertools
import json
import unittest
from pathlib import Path
import numpy as np
import torch
from concat_ops import geometry, source_order, reference
from model_graph import build_graph_model, evaluate_reference, input_specs, validate_graph
from fx_to_hecate import translate, Emitter, CipherValue
from candidate_trace import evaluate_tree
BASE=Path(__file__).parent


class Slots:
    """Plaintext diagnostic only; not an FHE backend."""
    def __init__(self,value):self.value=np.asarray(value,dtype=float)
    def __add__(self,other):return Slots(self.value+(other.value if isinstance(other,Slots) else other))
    def __mul__(self,other):return Slots(self.value*(other.value if isinstance(other,Slots) else other))
    def rotate(self,step):return Slots(np.roll(self.value,-step))


class ConcatTests(unittest.TestCase):
    def test_axis_mapping_exhaustive_small_equal_shapes(self):
        for rank in range(1,5):
            for shape in itertools.product((1,2),repeat=rank):
                for count in range(1,9):
                    if np.prod(shape)*count>8:continue
                    values=[(np.arange(np.prod(shape))+10*i).reshape(shape) for i in range(count)]
                    for axis in range(-rank,rank):
                        expected=np.concatenate(values,axis=axis)
                        out,order=source_order([shape]*count,axis)
                        actual=[values[b].reshape(-1)[i] for b,i in order]
                        np.testing.assert_array_equal(np.array(actual).reshape(out),expected)
                        np.testing.assert_array_equal(reference([v.tolist() for v in values],axis),expected)
        out,order=source_order([(1,1),(1,2),(1,1)],-1)
        self.assertEqual((out,order),((1,4),[(0,0),(1,0),(1,1),(2,0)]))

    def test_models_independent_inputs_and_all_scalar_slots(self):
        from run_concat_goldens import NAMES
        for name in (*NAMES,'concat-axis0-sensitive'):
            d=json.loads((BASE/'cases'/(name+'.json')).read_text())
            model,shape=build_graph_model(d);payload=translate(model,shape)
            for batch in range(4):
                arrays=[np.zeros(s['shape']) if batch==0 else
                        np.random.default_rng(140+batch*10+i).uniform(-1,1,s['shape'])
                        for i,s in enumerate(input_specs(d))]
                original=copy.deepcopy(d)
                given=arrays[0].tolist() if d['schema']==2 else {s['name']:a.tolist() for s,a in zip(input_specs(d),arrays)}
                expected=np.asarray(evaluate_reference(d,given))
                np.testing.assert_allclose(model(*[torch.from_numpy(a) for a in arrays]).detach().numpy(),expected,atol=1e-12,rtol=1e-12)
                output=evaluate_tree(payload['hecate_source'],payload['public_constants'],encrypted_inputs={
                    n:Slots(a.reshape(-1)) for n,a in zip(('x','y'),arrays)})
                if not isinstance(output,list):output=[output]
                # Check all four slots, not only output selector slot zero.
                for i,value in enumerate(output):np.testing.assert_allclose(value.value,np.repeat(expected[i],4),atol=1e-12,rtol=1e-12)
                self.assertEqual(d,original)

    def test_packed_mixed_and_scalar_neuron_conversion(self):
        emitter=Emitter()
        result=emitter.concat([CipherValue(('x',),(2,2),True),CipherValue(('a','b','c','d'),(2,2),False)],-1,'cat')
        self.assertEqual(result.shape,(2,4));self.assertFalse(result.packed)
        self.assertEqual(result.names[2:4],('a','b'))
        self.assertEqual(result.names[6:8],('c','d'))
        self.assertEqual(len(emitter.constants),4)
        second=Emitter()
        self.assertEqual(second.concat([CipherValue(('a',),(1,),False),CipherValue(('b','c'),(2,),False)],0,'c').names,('a','b','c'))
        self.assertEqual(second.lines,[])

    def test_axis0_observation_blindspot_and_separate_discriminator(self):
        supplied={'x':[[1.,2.],[3.,4.]],'y':[[5.,6.],[7.,8.]]}
        old=json.loads((BASE/'cases/concat-axis0.json').read_text())
        new=json.loads((BASE/'cases/concat-axis0-sensitive.json').read_text())
        for original,should_differ in ((old,False),(new,True)):
            wrong=copy.deepcopy(original)
            next(n for n in wrong['nodes'] if n['op']=='concat')['axis']=1
            self.assertEqual(evaluate_reference(original,supplied)!=evaluate_reference(wrong,supplied),should_differ)
        self.assertEqual(evaluate_reference(new,supplied),[204.,2.5])
        self.assertEqual(evaluate_reference(wrong,supplied),[196.,4.])

    def test_aliases_negative_axis_and_rejected_arguments(self):
        for fn in (torch.cat,torch.concat,torch.concatenate):
            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer('w',torch.tensor([[1.,0.,0.,0.,0.,0.,0.,0.],
                                                          [0.,0.,0.,0.,0.,0.,0.,1.]],dtype=torch.float64))
                def forward(self,x):
                    a=x.reshape(1,4)
                    return torch.nn.functional.linear(fn([a,a],dim=-2).reshape(8),self.w)
            payload=translate(M().eval(),[4])
            out=evaluate_tree(payload['hecate_source'],payload['public_constants'],
                              encrypted_inputs={'x':Slots([1.,2.,3.,4.])})
            np.testing.assert_array_equal(out[0].value,[1.]*4)
            np.testing.assert_array_equal(out[1].value,[4.]*4)
        for shapes,axis in [([(2,2),(1,4)],0),([(4,),(4,)],True),([(4,),(4,)],1),
                            ([(4,),(4,)],-2),([(4,)]*3,0),([],0),([(0,),(4,)],0),
                            ([(2,2),(4,)],0),([(4,),(4,)],'0')]:
            with self.subTest(shapes=shapes,axis=axis),self.assertRaises(ValueError):geometry(shapes,axis)
        d=json.loads((BASE/'cases/concat-vector.json').read_text())
        d['nodes'][-1]['inputs']=['a1','a']
        with self.assertRaisesRegex(ValueError,'ciphertext'):validate_graph(d)


if __name__=='__main__':unittest.main()
