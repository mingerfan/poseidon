"""S1 §9 stage observability and the compiler's minimal shared-value fanout."""

import json
import os
from pathlib import Path
import unittest

from pt_agent import toolchain
from pt_agent.experiments.s1_managed import compare_ir
from pt_agent.experiments.s1_stages import minimal_input, pipeline
from pt_agent.manage import manage, emit, read
from pt_agent.manage.ir import Function, Node, Type
from pt_agent.paths import pt_result
from pt_agent.tests._artifacts import COMPILER, require


PROOF = pt_result('s1-input-proof-20261026T045526810251/shared-value')
OUTPUT_DIR = None


class StageUnitTests(unittest.TestCase):
    def test_registered_options_derive_the_current_profile(self):
        p = toolchain.management_profile()
        self.assertIn('waterline='+str(p['waterline']), pipeline('a',p))
        self.assertIn('output-val='+str(p['output_val']), pipeline('a',p))
        self.assertIn('depth-dp=true', pipeline('a',p))
        self.assertNotIn('early-modswitch', pipeline('a',p))
        self.assertIn('early-modswitch', pipeline('b',p))
        self.assertTrue(pipeline('c',p).endswith(','+toolchain.EARTH_TAIL_PASSES+')'))

    def test_optional_stage_observer_preserves_the_final_result(self):
        p = toolchain.management_profile()
        stages = []
        observed = manage(minimal_input(True), p,
                          stage_callback=lambda name,f:stages.append((name,f.summary())))
        plain = manage(minimal_input(True), p)
        self.assertEqual([name for name,_ in stages], ['a','b'])
        self.assertEqual(stages[-1][1], observed.summary())
        self.assertTrue(compare_ir(plain, observed)['equal'])


@require(COMPILER, PROOF/'compiler-a/STRUCTURE.json', PROOF/'compiler-b/STRUCTURE.json')
class NativeSharedFanoutTests(unittest.TestCase):
    def test_shared_fanout_stage_a_and_b_match_the_compiler(self):
        if OUTPUT_DIR is None:
            self.skipTest('Run with the S1 unit entry to preserve native logs')
        root = Path(OUTPUT_DIR)
        root.mkdir()
        p = toolchain.management_profile()

        def compare(stage, function):
            directory = root/stage
            directory.mkdir()
            records = json.loads((PROOF/('compiler-'+stage)/'STRUCTURE.json').read_text())
            nodes = [Node(r['id'],r['kind'],r['args'],Type(True,r['scale'],r['level']),r['attrs']) for r in records]
            expected = Function('golden',[Type(True,p['waterline'],0)],nodes,[nodes[-1].id],
                [nodes[-1].type],dict(arg_scale=[p['waterline']],init_level=p['bootstrapLevelUpperBound'],
                                     res_scale=[p['waterline']]),p)
            emit(function,directory/'managed.mlir')
            command = [str(toolchain.host_paths()['hecate_opt']),str(directory/'managed.mlir'),
                '--ckks-config='+str(toolchain.BOOT_CONFIG_DIR/'compiler-config.json'),
                '--mlir-disable-threading','--verify-each','--pass-pipeline=builtin.module(canonicalize)',
                '-o',str(directory/'normalized.earth.mlir')]
            status = toolchain.trusted_run(directory,command,directory/'canonicalize.log',
                env_extra=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
            self.assertEqual(status['exit'],0,status['log_tail'])
            actual = read(directory/'normalized.earth.mlir',p)
            comparison = compare_ir(expected,actual)
            (directory/'COMPARISON.json').write_text(json.dumps(comparison,indent=2)+'\n')
            toolchain.prune(directory)
            self.assertTrue(comparison['equal'],comparison)
        manage(minimal_input(True),p,stage_callback=compare)
