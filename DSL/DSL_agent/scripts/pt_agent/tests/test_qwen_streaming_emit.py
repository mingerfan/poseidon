"""Tiny trusted source/storage checks; no frontend, model or native execution."""

import ast
import gc
import json
import os
from pathlib import Path
import tempfile
import unittest
import weakref


class StreamingEmitChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name]='1'
        import numpy as np
        from pt_agent.backends import CountBackend
        from pt_agent.backends.streaming_emit import ConstantWriter,ConstantReader,StreamingEmitBackend
        from pt_agent.backends.streaming_mock import StreamingMockBackend
        from pt_agent.kernels.streaming_linear import streaming_linear
        from pt_agent.kernels.layout import multi_flat_layout,layout_blocks
        cls.np,cls.Writer,cls.Reader,cls.Emit=np,ConstantWriter,ConstantReader,StreamingEmitBackend
        cls.Mock,cls.Count,cls.linear=StreamingMockBackend,CountBackend,staticmethod(streaming_linear)
        cls.layout,cls.blocks=staticmethod(multi_flat_layout),staticmethod(layout_blocks)

    def test_sparse_dense_and_signed_zero_roundtrip_without_retained_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'store'
            writer=self.Writer(directory,16)
            sparse=self.np.zeros(16);sparse[3]=.25;sparse[7]=-0.
            dense=self.np.arange(16,dtype=self.np.float64)+.125
            writer.add('k0',sparse);writer.add('k1',dense);writer.add('k0',sparse)
            writer.close()
            self.assertEqual(len(writer.names),2)
            self.assertLess(writer.payload_bytes,writer.logical_bytes)
            reader=self.Reader(directory)
            try:
                for name,original in [('k0',sparse),('k1',dense)]:
                    loaded=reader.load(name)
                    self.np.testing.assert_array_equal(loaded,original)
                    self.np.testing.assert_array_equal(self.np.signbit(loaded),self.np.signbit(original))
                    self.assertFalse(loaded.flags.writeable)
                self.assertEqual(reader.records['k0']['encoding'],'indexed64')
                self.assertEqual(reader.records['k1']['encoding'],'dense64')
            finally:reader.close()

    def test_storage_allowance_rejects_before_partial_payload_or_index(self):
        from pt_agent.backends import BackendError
        with tempfile.TemporaryDirectory() as temporary:
            writer=self.Writer(Path(temporary)/'store',16,max_bytes=1)
            try:
                with self.assertRaisesRegex(BackendError,'allowance'):
                    writer.add('k0',self.np.ones(16))
                self.assertEqual(writer.payload_bytes,0)
                self.assertEqual(writer.index_bytes,0)
                self.assertEqual(writer.names,set())
            finally:writer.close()

    def test_reader_rejects_outside_ranges_and_duplicate_names(self):
        from pt_agent.backends import BackendError
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'store'
            writer=self.Writer(directory,8);writer.add('k0',self.np.ones(8));writer.close()
            index=directory/'constants-index.jsonl';original=index.read_text()
            record=json.loads(original);record['offset']=100000
            index.write_text(json.dumps(record)+'\n')
            with self.assertRaisesRegex(BackendError,'ranges'):self.Reader(directory)
            index.write_text(original+original)
            with self.assertRaisesRegex(BackendError,'ranges'):self.Reader(directory)

    def _replay(self, source, reader, backend, inputs):
        """Interpret only the finite arithmetic statements emitted by our backend."""
        function=next(node for node in ast.walk(ast.parse(source)) if isinstance(node,ast.FunctionDef))
        env=dict(zip([arg.arg for arg in function.args.args],inputs))
        def value(node):
            if isinstance(node,ast.Name):return env[node.id]
            if isinstance(node,ast.Constant):return node.value
            if isinstance(node,ast.List):return [value(item) for item in node.elts]
            if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -value(node.operand)
            if isinstance(node,ast.BinOp):
                left,right=value(node.left),value(node.right)
                if isinstance(node.op,ast.Add):return left+right
                if isinstance(node.op,ast.Sub):return left-right
                if isinstance(node.op,ast.Mult):return left*right
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                function=node.func
                if isinstance(function.value,ast.Name) and function.value.id=='C' and function.attr=='load':
                    return reader.load(value(node.args[0]))
                if isinstance(function.value,ast.Name) and function.value.id=='hc':
                    if function.attr=='Plain':return backend.const(value(node.args[0]))
                    if function.attr=='bootstrap':return value(node.args[0]).bootstrap()
                if function.attr=='rotate':return value(function.value).rotate(value(node.args[0]))
            raise AssertionError('Unexpected emitted arithmetic: '+ast.dump(node))
        for statement in function.body:
            if isinstance(statement,ast.Assign):env[statement.targets[0].id]=value(statement.value)
            elif isinstance(statement,ast.Return):return value(statement.value)
            else:self.fail('Unexpected emitted statement')

    def test_original_embedding_schedule_emits_count_matching_source_and_strict_replay(self):
        from pt_agent.kernels.encrypted_generation import embedding_from_selection
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'emit'
            emitter=self.Emit(16,directory=directory,frontend='/unused/frontend.py')
            count=self.Count(16)
            source=self.np.sin(self.np.arange(19,dtype=self.np.float64))
            embedding=self.np.cos(self.np.arange(19*5,dtype=self.np.float64)).reshape(19,5)
            layout=self.layout(16,len(source));packed=[]
            for start,stop,_ in self.blocks(layout):
                vector=self.np.zeros(16);vector[:stop-start]=source[start:stop];packed.append(vector)
            for backend in (emitter,count):
                selection=dict(one_hot=[backend.input(index) for index in range(len(packed))],
                    layout=layout,metadata=dict(vocabulary=len(source)))
                result,output=embedding_from_selection(backend,selection,embedding)
                backend.output(result)
            emitted=emitter.stats();expected=count.stats()
            for key in ('counts','ops_total','distinct_consts'):self.assertEqual(emitted[key],expected[key])
            path=emitter.write();tree=ast.parse(path.read_text())
            self.assertEqual(sum(isinstance(node,ast.FunctionDef) for node in ast.walk(tree)),1)
            self.assertEqual(emitted['retained_constant_arrays'],0)
            self.assertEqual(emitted['identity_cache_entries'],0)
            mock=self.Mock(16);mock.bind(packed)
            reader=self.Reader(directory/'constant-store')
            try:observed=self._replay(path.read_text(),reader,mock,[mock.input(index) for index in range(len(packed))])
            finally:reader.close()
            observed=observed.value[:5] if not isinstance(observed,list) else observed[0].value[:5]
            reference=source@embedding
            self.assertTrue(self.np.all(self.np.abs(reference-observed)<=1e-12+1e-12*self.np.abs(reference)))
            self.assertFalse(emitted['native_trace_executions'])

    def test_readonly_operand_payload_is_released_after_storage_and_arithmetic(self):
        with tempfile.TemporaryDirectory() as temporary:
            backend=self.Emit(16,directory=Path(temporary)/'emit',frontend='/unused/frontend.py')
            vector=self.np.zeros(16);vector[3]=.75;vector.flags.writeable=False
            original=weakref.ref(vector)
            plain=backend.const(vector);payload=weakref.ref(plain.data)
            output=backend.input(0)*plain
            backend.output([output]);backend.write()
            del vector,plain;gc.collect()
            self.assertIsNone(original());self.assertIsNone(payload())
            self.assertFalse(backend._constant_objects)
            self.assertTrue(all(value.data is None for value in backend._constants.values()))
            self.assertEqual(backend.stats()['stored_constants'],1)


if __name__=='__main__':unittest.main()
