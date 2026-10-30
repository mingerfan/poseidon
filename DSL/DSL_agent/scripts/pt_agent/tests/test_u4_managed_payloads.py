"""U4 §7.6 shared dense payloads and checked structural reconstruction."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from pt_agent import toolchain
from pt_agent.export.package import frontend_preprocess_template,frontend_structure_template
from pt_agent.manage import RecordingBackend,emit_raw,emit,manage
from pt_agent.manage.earth_emit import constant_sizes
from pt_agent.manage.mlir_read import read


class SharedPayloadTests(unittest.TestCase):
    def recorded(self):
        b = RecordingBackend(1024)
        b.op_context = 7
        x = b.input(0)
        k = b.const(np.linspace(-1.,2.,1024))
        y = x*k
        for _ in range(6):
            y = y+k
        b.output([y])
        return b,k

    def test_alias_written_once_private_nodes_and_owner_preserved(self):
        b,k = self.recorded()
        with TemporaryDirectory() as tmp:
            path = emit_raw(b,Path(tmp)/'raw.mlir',toolchain.management_profile())
            text = path.read_text()
            self.assertEqual(text.count('dense<'),1)
            self.assertEqual(text.count('"earth.constant"'),7)
            self.assertEqual(text.count('value = #pt'+k.name),7)
            self.assertEqual(text.count('ptdsl-const-'+k.name),7)
            f = read(path,{})
            constants = [n for n in f.nodes if n.kind=='constant']
            self.assertEqual(len(f.nodes),len(b.nodes))
            self.assertTrue(all(n.data is constants[0].data for n in constants))
            self.assertTrue(all(n.const_key==k.name and n.op_id==7 for n in constants))
            np.testing.assert_array_equal(constants[0].data,k.data)

    def test_negative_plain_constant_uses_existing_const_identity(self):
        b = RecordingBackend(1024)
        x = b.input(0)
        k = b.const([2.])
        b.output([x-k])
        node = b.nodes[0]
        self.assertEqual(node.const_key,b.const([-2.]).name)
        self.assertNotEqual(node.const_key,k.name)
        np.testing.assert_array_equal(node.data,np.full(1024,-2.))

    def test_structure_restores_shared_table_arrays_and_checks_rms(self):
        b,k = self.recorded()
        profile = toolchain.management_profile()
        with TemporaryDirectory() as tmp:
            path = emit_raw(b,Path(tmp)/'raw.mlir',profile)
            text = path.read_text()
            # Model the native elision printer; NameLoc/rms/edges stay intact.
            text = '\n'.join(line for line in text.splitlines() if not line.startswith('#pt'))
            text = text.replace('value = #pt'+k.name,'value = dense<"__elided__"> : tensor<1024xf64>')
            path.write_text(text)
            f = read(path,profile,constants={k.name:k.data})
            constants = [n for n in f.nodes if n.kind=='constant']
            self.assertTrue(all(n.data is k.data for n in constants))
            self.assertEqual([n.args for n in f.nodes],[n.args for n in b.nodes])
            path.write_text(text.replace('ptdsl-const-'+k.name,'unnamed'))
            with self.assertRaisesRegex(ValueError,'lacks ptdsl-const'):
                read(path,profile,constants={k.name:k.data})
            path.write_text(text)
            with self.assertRaisesRegex(ValueError,'rms_var differs'):
                read(path,profile,constants={k.name:k.data*2.})

    def test_private_encoding_estimate_uses_payload_type_and_rms(self):
        b,k = self.recorded()
        f = manage(b,toolchain.management_profile())
        size = constant_sizes(f)
        self.assertEqual(size['constant_ops'],7)
        self.assertEqual(size['distinct_payloads'],1)
        combos = {(n.const_key,n.type) for n in f.nodes if n.kind=='constant'}
        self.assertEqual(size['encoding_combinations'],len(combos))
        self.assertEqual(size['estimated_cst_bytes'],len(combos)*1024*8)
        with TemporaryDirectory() as tmp:
            metrics = {}
            path = emit(f,Path(tmp)/'managed.mlir',metrics=metrics)
            self.assertEqual(path.read_text().count('dense<'),1)
            self.assertEqual(metrics['estimated_cst_bytes'],size['estimated_cst_bytes'])

    def test_bytecode_preprocess_and_no_pass_structure_commands(self):
        command = toolchain.frontend_preprocess_argv('dacapo-depth-dp','qwen48',Path('.'),bytecode=True)
        self.assertIn('--emit-bytecode',command)
        self.assertEqual(command[-1],'/out/preprocessed.mlirbc')
        structure = toolchain.frontend_structure_argv('dacapo-depth-dp','qwen48',Path('.'))
        self.assertIn('--mlir-elide-elementsattrs-if-larger=1',structure)
        self.assertFalse(any(arg.startswith('--pass-pipeline=') for arg in structure))
        substitutions = dict(hecate_opt='/hecate-opt',input='/out/raw.earth.mlir',package='/out',output_dir='/out')
        for source,template in ((command,frontend_preprocess_template(bytecode=True)),
                                (structure,frontend_structure_template())):
            expected = [a.replace('/out/boot-compiler-config.json','/out/config/compiler-config.json') for a in source]
            self.assertEqual([a.format(**substitutions) for a in template],expected)

    def test_bounded_plan_records_preserve_all_s1_state_fields(self):
        from pt_agent.experiments.u4_bert_managed import states
        from pt_agent.experiments.s0_pipeline_spike import step_states
        plan = dict(values=[dict(id='x',kind='ciphertext',level=5,scale_log2=43,components=2,ntt=True),
                            dict(id='y',kind='ciphertext',level=4,scale_log2=43,components=2,ntt=True)],
                    initialization=[],execution=[dict(op='rotate',inputs=['x'],output='y',attrs=dict(step=3))],
                    finalization=[])
        self.assertEqual(list(states(plan)),step_states(plan))


if __name__=='__main__':
    unittest.main()
