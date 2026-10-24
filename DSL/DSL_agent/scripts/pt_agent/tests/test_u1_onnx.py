"""Small ONNX semantics checks; executed only in the isolated intake venv."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

from pt_agent.intake.onnx_frontend import import_graph
from pt_agent.intake.graph_exec import forward


@unittest.skipUnless(Path(sys.executable).parent.parent.name == 'intake-2.0.1-cpu', 'requires independent intake venv')
class ONNXSemanticsTests(unittest.TestCase):
    def test_static_broadcast_shape_is_reinferred_after_folding(self):
        import onnx
        from onnx import helper, numpy_helper
        with tempfile.TemporaryDirectory() as directory:
            values={'index':np.zeros((1,4),dtype=np.int64),'cond':np.array([False,False]),
                    'ones':np.array([1,1],dtype=np.int64),'target':np.array([1,4],dtype=np.int64),
                    'table':np.array([[3.,4.],[5.,6.]],dtype=np.float32)}
            nodes=[helper.make_node('Where',['cond','ones','target'],['shape']),
                   helper.make_node('Expand',['index','shape'],['indices']),
                   helper.make_node('Gather',['table','indices'],['offset'],axis=0),
                   helper.make_node('Add',['x','offset'],['added']),
                   helper.make_node('Relu',['added'],['y'])]
            graph=helper.make_graph(nodes,'constant_broadcast',
                [helper.make_tensor_value_info('x',onnx.TensorProto.FLOAT,[1,4,2])],
                [helper.make_tensor_value_info('y',onnx.TensorProto.FLOAT,[1,4,2])],
                [numpy_helper.from_array(value,key) for key,value in values.items()])
            path=Path(directory)/'broadcast.onnx'
            onnx.save(helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]),path)
            x=np.arange(8,dtype=np.float32).reshape(1,4,2)
            imported=import_graph(path,{'x':x})
            self.assertEqual(imported.tensors['added'].shape,(1,4,2))
            np.testing.assert_array_equal(forward(imported,{'x':x})['y'],x+np.array([3.,4.]))

    def test_static_pad_arguments_and_omitted_constant_value(self):
        import onnx
        from onnx import helper, numpy_helper
        with tempfile.TemporaryDirectory() as directory:
            nodes = [helper.make_node('Constant', [], ['before'], value=numpy_helper.from_array(np.array([0, 1], dtype=np.int64))),
                     helper.make_node('Constant', [], ['after'], value=numpy_helper.from_array(np.array([0, 2], dtype=np.int64))),
                     helper.make_node('Concat', ['before', 'after'], ['pads'], axis=0),
                     helper.make_node('Pad', ['x', 'pads', ''], ['padded']),
                     helper.make_node('Relu', ['padded'], ['y'])]
            graph = helper.make_graph(nodes, 'static_padding',
                [helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 3])],
                [helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 6])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
            path = Path(directory)/'pad.onnx'; onnx.save(model, path)
            x = np.array([[-1., 2., 3.]], dtype=np.float32)
            imported = import_graph(path, {'x': x})
            np.testing.assert_array_equal(forward(imported, {'x': x})['y'], [[0., 0., 2., 3., 0., 0.]])
            self.assertEqual(imported.intake_stats['folded_nodes'], 3)

    def test_conv_auto_pad_modes(self):
        import onnx
        from onnx import helper, numpy_helper
        for mode in ('SAME_UPPER', 'SAME_LOWER', 'VALID'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                x = np.arange(30, dtype=np.float32).reshape(1, 1, 5, 6)/30
                w = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)/9
                shape = [1, 1, 2, 2] if mode == 'VALID' else [1, 1, 3, 3]
                node = helper.make_node('Conv', ['x', 'w'], ['y'], auto_pad=mode, strides=[2, 2])
                graph = helper.make_graph([node], 'padding', [helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, x.shape)],
                    [helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, shape)], [numpy_helper.from_array(w, 'w')])
                model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
                path = Path(directory)/'conv.onnx'; onnx.save(model, path)
                imported = import_graph(path, {'x': x})
                actual = forward(imported, {'x': x})['y']
                padding = (0, 0, 0, 0) if mode == 'VALID' else (0, 1, 1, 1) if mode == 'SAME_UPPER' else (1, 0, 1, 1)
                expected = torch.nn.functional.conv2d(torch.nn.functional.pad(torch.from_numpy(x).double(), padding),
                                                      torch.from_numpy(w).double(), stride=2).numpy()
                np.testing.assert_array_equal(actual, expected)

    def test_inference_dropout_is_identity(self):
        import onnx
        from onnx import helper
        with tempfile.TemporaryDirectory() as directory:
            value = helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 4])
            output = helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 4])
            model = helper.make_model(helper.make_graph([helper.make_node('Dropout', ['x'], ['y'])], 'drop', [value], [output]),
                                      opset_imports=[helper.make_opsetid('', 17)])
            path = Path(directory)/'drop.onnx'; onnx.save(model, path)
            x = np.arange(4, dtype=np.float32).reshape(1, 4)
            graph = import_graph(path, {'x': x})
            np.testing.assert_array_equal(forward(graph, {'x': x})[graph.outputs[0]], x)


if __name__ == '__main__':
    unittest.main()
