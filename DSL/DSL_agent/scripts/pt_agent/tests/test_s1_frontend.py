"""S1 §10 unmanaged Earth, strict parsing and the single preprocessing command."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from pt_agent import toolchain
from pt_agent.export.package import frontend_preprocess_template
from pt_agent.manage import RecordingBackend, emit_raw
from pt_agent.manage.earth_emit import RMSCalculator
from pt_agent.manage.ir import Type
from pt_agent.manage.mlir_read import read, load_graph, RAW_OPS


GENERIC = '''"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>, sym_name = "_hecate_unit"}> ({
  ^bb0(%arg0: tensor<1x!earth.ci<0 * 0>>):
    %0 = "earth.negate"(%arg0) : (tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>> loc(#loc2)
    "func.return"(%0) : (tensor<1x!earth.ci<0 * 0>>) -> ()
  }) : () -> ()
}) : () -> ()
#loc1 = loc("ptdsl-op-44":9:0)
#loc2 = loc(fused[#loc1])
'''


class FrontendUnitTests(unittest.TestCase):
    def test_raw_builder_types_payload_and_rms(self):
        b = RecordingBackend(1024)
        x = b.input(0)
        data = np.linspace(-1,2,1024)
        b.output([x*b.const(data)])
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'raw.mlir'
            emit_raw(b,path,toolchain.management_profile())
            f = read(path,{},allowed=RAW_OPS)
            self.assertEqual(f.inputs,[Type(True,0,0)])
            self.assertEqual(f.result_types,[Type(True,0,0)])
            self.assertEqual(f.attrs,{})
            constant = next(n for n in f.nodes if n.kind=='constant')
            self.assertEqual(constant.type,Type(False,0,0))
            np.testing.assert_array_equal(constant.data,data)
            self.assertEqual(constant.attrs['rms_var'],RMSCalculator('fma')(data))

    def test_generic_function_return_and_location_aliases(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'raw.mlir'
            path.write_text(GENERIC)
            f = read(path,{},allowed=RAW_OPS)
            self.assertEqual(f.name,'unit')
            self.assertEqual(f.outputs,[0])
            self.assertEqual(f.nodes[0].args,[-1])
            self.assertEqual((f.nodes[0].op_id,f.nodes[0].lineno),(44,9))

    def test_ssa_sharing_and_operand_slots_survive_reading(self):
        source = GENERIC.replace('"earth.negate"(%arg0)', '"earth.mul"(%arg0, %arg0)')
        source = source.replace(': (tensor<1x!earth.ci<0 * 0>>) -> tensor',
            ': (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor')
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'raw.mlir'
            path.write_text(source)
            b = load_graph(path,1024)
            self.assertEqual(b.nodes[0].args,[-1,-1])
            self.assertEqual(len(b._inputs),1)
            self.assertEqual(b._outputs[0].nid,0)

    def test_unknown_operations_are_rejected(self):
        for op in ('earth.unknown','arith.addf','func.call'):
            with self.subTest(op=op), TemporaryDirectory() as temporary:
                path = Path(temporary)/'raw.mlir'
                path.write_text(GENERIC.replace('earth.negate',op))
                with self.assertRaisesRegex(ValueError,'unsupported'):
                    read(path,{},allowed=RAW_OPS)

    def test_managed_operations_are_rejected_as_unmanaged_input(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'raw.mlir'
            path.write_text(GENERIC.replace('earth.negate','earth.rescale'))
            with self.assertRaisesRegex(ValueError,'rescale'):
                load_graph(path,1024)

    def test_preprocessing_template_uses_the_single_source(self):
        commands = toolchain.frontend_preprocess_argv('dacapo-depth-dp','qwen48',Path('.'))
        self.assertIn('--pass-pipeline=builtin.module(cse,canonicalize,func.func(privatize-constant),canonicalize)',commands)
        substitutions = dict(hecate_opt='/hecate-opt',input='/out/raw.earth.mlir',
                             package='/out',output_dir='/out')
        actual = [a.format(**substitutions) for a in frontend_preprocess_template()]
        expected = [a.replace('/out/boot-compiler-config.json','/out/config/compiler-config.json') for a in commands]
        self.assertEqual(actual,expected)
