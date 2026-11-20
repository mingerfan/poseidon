"""Small all-branch streaming projections; no original model or data reads."""

import gc
import os
import unittest
import weakref


class StreamingProjectionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name]='1'
        import numpy as np
        from pt_agent.backends import CountBackend,MockBackend
        from pt_agent.backends.streaming_count import StreamingCountBackend
        from pt_agent.backends.streaming_mock import StreamingMockBackend
        from pt_agent.kernels import transformer,streaming_linear,slots
        from pt_agent.kernels.layout import multi_flat_layout,TokenLayout,layout_blocks
        cls.np,cls.Count,cls.StreamCount,cls.Mock,cls.StreamMock=np,CountBackend,StreamingCountBackend,MockBackend,StreamingMockBackend
        cls.original,cls.stream,cls.scopes=transformer,streaming_linear,slots
        cls.flat,cls.Token,cls.blocks=staticmethod(multi_flat_layout),TokenLayout,staticmethod(layout_blocks)

    def bind(self,backend,array,layout):
        packed=[]
        for start,stop,_ in self.blocks(layout):
            vector=self.np.zeros(backend.slots);vector[:stop-start]=array.reshape(-1)[start:stop];packed.append(vector)
        if isinstance(backend,self.Mock):backend.bind(packed)
        return [backend.input(index) for index in range(len(packed))]

    def unpack(self,result,layout):
        values=self.original.values(result,layout)
        return self.np.concatenate([value.value[:stop-start] for value,(start,stop,_) in zip(values,self.blocks(layout))])

    def test_flat_multiblock_token_and_tall_schedules_match_original_and_numpy(self):
        for size,hidden,tokens in ((5,3,None),(3,20,None),(5,9,None),(19,11,None),(5,3,3),(3,20,3),(19,11,2)):
            with self.subTest(size=size,hidden=hidden,tokens=tokens):
                source=self.np.sin(self.np.arange((tokens or 1)*size,dtype=self.np.float64)*.3)
                if tokens:source=source.reshape(tokens,size)
                weight=self.np.cos(self.np.arange(hidden*size,dtype=self.np.float64).reshape(hidden,size)*.31)
                bias=self.np.linspace(-.1,.2,hidden)
                layout=self.Token(16,tokens,size) if tokens else self.flat(16,size)
                original,stream=self.Mock(16),self.StreamMock(16)
                out0,lay0=self.original.linear(original,self.bind(original,source,layout),layout,weight,bias)
                out1,lay1=self.stream.streaming_projection(stream,self.bind(stream,source,layout),layout,weight,bias)
                self.assertEqual(lay0,lay1)
                reference=self.unpack(out0,lay0);observed=self.unpack(out1,lay1)
                direct=(source@weight.T+bias).reshape(-1)
                self.assertTrue(self.np.all(self.np.abs(observed-reference)<=1e-12+1e-12*self.np.abs(reference)))
                self.assertTrue(self.np.all(self.np.abs(observed-direct)<=1e-12+1e-12*self.np.abs(direct)))
                self.assertFalse(stream._constant_objects)
                self.assertTrue(all(value.data is None for value in stream._constants.values()))

    def test_all_count_branches_preserve_original_counts(self):
        for size,hidden,tokens in ((3,20,None),(19,11,None),(3,20,3),(5,3,3)):
            with self.subTest(size=size,hidden=hidden,tokens=tokens):
                layout=self.Token(16,tokens,size) if tokens else self.flat(16,size)
                source=self.np.ones((tokens or 1)*size)
                weight=self.np.arange(hidden*size,dtype=self.np.float64).reshape(hidden,size)+.1
                original,stream=self.Count(16),self.StreamCount(16)
                self.original.linear(original,self.bind(original,source,layout),layout,weight)
                self.stream.streaming_projection(stream,self.bind(stream,source,layout),layout,weight)
                self.assertEqual(original.stats(),stream.stats())
                self.assertFalse(stream._constant_objects)

    def test_trusted_binding_and_outer_cache_restore_even_on_failure(self):
        from pt_agent.kernels.layout import KernelError
        backend=self.StreamCount(16);layout=self.flat(16,5)
        values=self.bind(backend,self.np.ones(5),layout)
        previous=(backend,{},{});token=self.scopes._current.set(previous)
        original=self.original.linear
        try:
            with self.assertRaises(KernelError):
                with self.stream.streaming_transformer_projections():
                    self.assertIs(self.original.linear,self.stream.streaming_projection)
                    self.original.linear(backend,values,layout,self.np.ones((3,5)),algo='invalid')
            self.assertIs(self.original.linear,original)
            self.assertIs(self.scopes._current.get(),previous)
            self.assertEqual(previous[1],{})
        finally:self.scopes._current.reset(token)

    def test_readonly_constant_does_not_retain_its_original_array(self):
        backend=self.StreamCount(16)
        value=self.np.arange(16,dtype=self.np.float64);value.flags.writeable=False
        reference=weakref.ref(value)
        backend.const(value);del value;gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(backend._constant_objects)
        self.assertTrue(all(value.data is None for value in backend._constants.values()))


if __name__=='__main__':unittest.main()
