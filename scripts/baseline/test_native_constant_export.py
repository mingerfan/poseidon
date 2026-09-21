"""Real sequential MLIR pass reuse regression; no model API or FHE claim."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from workspace_paths import ROOT, WORK

COMPILER = WORK / 'build-dacapo/hecate-18.1.2-nix/bin/hecate-opt'


@unittest.skipUnless(os.environ.get('IN_NIX_SHELL') and COMPILER.is_file(),
                     'requires the already built native Hecate compiler in Nix')
class NativeConstantExportTests(unittest.TestCase):
    def test_reused_pass_keeps_output_prefix_immutable(self):
        # No constants are needed to reproduce the path mutation. Keep all files
        # on failure, so a concatenated filename remains inspectable evidence.
        output = Path(tempfile.mkdtemp(prefix='constant-export-regression-', dir=WORK / 'results'))
        source = output / 'functions.mlir'
        source.write_text('module {\n' + ''.join(
            f'  func.func @{name}() {{ return }}\n' for name in ('first', 'second', 'third')) + '}\n')
        command = [str(COMPILER), str(source), '--mlir-disable-threading',
                   '--ckks-config=' + str(ROOT / 'third_party/dacapo/config.json'),
                   '--pass-pipeline=builtin.module(func.func(elide-constant{name=' + str(output) + '/}))',
                   '-o', str(output / 'lowered.mlir')]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        (output / 'compiler.log').write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, str(output) + '\n' + result.stderr)
        self.assertEqual(sorted(p.name for p in output.glob('*.cst')),
                         ['first.cst', 'second.cst', 'third.cst'], str(output))
        for name in ('first', 'second', 'third'):
            self.assertEqual((output / (name + '.cst')).read_bytes(), struct.pack('<q', 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
