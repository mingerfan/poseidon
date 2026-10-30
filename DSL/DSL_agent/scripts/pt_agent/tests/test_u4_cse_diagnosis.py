"""U4 §7.7 ordered value numbering and non-executable structural parsing."""

from pathlib import Path
import tempfile
import unittest

from pt_agent.experiments.u4_cse_diagnosis import numbered_dag
from pt_agent.manage.mlir_read import read, RAW_OPS


class NumberingTests(unittest.TestCase):
    def test_rotation_attributes_and_ordered_operands(self):
        b = numbered_dag(8,reuse=True)
        x,y = b.input(0),b.input(1)
        self.assertEqual(x.rotate(1).nid,x.rotate(1).nid)
        self.assertNotEqual(x.rotate(1).nid,x.rotate(2).nid)
        self.assertNotEqual((x*y).nid,(y*x).nid)
        self.assertEqual((x*y).nid,(x*y).nid)

    def test_plain_identity_and_mul_cp(self):
        b = numbered_dag(8,reuse=True)
        x = b.input(0)
        k0,k0_again,k1 = b.const([2.]),b.const([2.]),b.const([3.])
        self.assertEqual(k0.name,k0_again.name)
        self.assertEqual((x*k0).nid,(x*k0_again).nid)
        self.assertNotEqual((x*k0).nid,(x*k1).nid)

    def test_raw_graph_keeps_repeated_operations(self):
        b = numbered_dag(8,reuse=False)
        x = b.input(0)
        self.assertNotEqual(x.rotate(1).nid,x.rotate(1).nid)
        self.assertEqual(b.stats()['counts']['rot'],2)


class StructuralReaderTests(unittest.TestCase):
    def test_structure_has_types_edges_rms_and_owner_without_payload(self):
        text = '''module {
  func.func @_hecate_unit(%arg0: tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>> {
    %v0 = "earth.constant"() {rms_var = 0x3FF0000000000000 : f64, value = dense_resource<__elided__> : tensor<8xf64>} : () -> tensor<1x!earth.pl<0 * 0>> loc(#loc1)
    %v1 = "earth.mul"(%arg0, %v0) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>> loc(#loc1)
    return %v1 : tensor<1x!earth.ci<0 * 0>>
  }
}
#loc1 = loc("ptdsl-op-4":9:0)
'''
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'structure.mlir'
            path.write_text(text)
            f = read(path,{},allowed=RAW_OPS,structure_only=True)
            self.assertIsNone(f.nodes[0].data)
            self.assertEqual(f.nodes[0].attrs['rms_var'],1.)
            self.assertEqual(f.nodes[1].args,[-1,0])
            self.assertEqual((f.nodes[1].op_id,f.nodes[1].lineno),(4,9))
            with self.assertRaisesRegex(ValueError,'unsupported dense'):
                read(path,{},allowed=RAW_OPS)


if __name__=='__main__':
    unittest.main()
