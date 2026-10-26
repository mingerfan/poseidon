"""S1 management, source ownership, portable packaging and default-mode contracts."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from pt_agent import toolchain
from pt_agent.backends import DagBackend
from pt_agent.dsl.ir import canonical
from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.dsl.typecheck import check as typecheck
from pt_agent.export import export_package
from pt_agent.export.package import managed_compile_template
from pt_agent.manage import RecordingBackend, manage, check, emit, read, ManagementError
from pt_agent.manage.greedy import infer
from pt_agent.manage.earth_emit import rms
from pt_agent.placement import depth_dp
from pt_agent.tests._artifacts import COMPILER, require


SMALL = ('@fhe.program(slots=8192)\ndef square(x: Cipher[1,4]):\n'
         ' a = pack(x, layout="flat")\n b = act(a, fn="square", method="exact")\n return b\n')


def record_case(name, profile=None):
    profile = toolchain.management_profile() if profile is None else profile
    b = RecordingBackend(1024, locations={12: SimpleNamespace(id=12, lineno=7)})
    b.op_context = 12
    x = b.input(0)
    if name == 'add':
        y = x+x
    elif name == 'chain':
        y = x
        for _ in range(9):
            y = y*y
    elif name == 'residual':
        y = x
        for _ in range(9):
            y = y*y
        y = y+x
    elif name == 'constants':
        k = b.const(np.arange(1024, dtype=float)/1024+.5)
        y = x*k
        for _ in range(5):
            y = y*y
        y = y+k
    elif name == 'no-final-mul':
        y = (x*x).rotate(3)+x
    else:
        raise ValueError(name)
    b.output([y])
    return b, manage(b, profile, name=name.replace('-', '_'))


class HeaderTests(unittest.TestCase):
    def test_default_and_explicit_compiler_preserve_text(self):
        p = parse(SMALL)
        text = print_program(p)
        self.assertNotIn('manage=', text)
        self.assertEqual(text, print_program(parse(SMALL.replace('slots=8192', 'slots=8192, manage="compiler"'))))
        self.assertNotIn('manage', canonical(p)['header'])

    def test_dsl_roundtrip_and_unsupported_combinations(self):
        p = parse(SMALL.replace('slots=8192', 'slots=8192, manage="dsl"'))
        self.assertTrue(typecheck(p, {}).ok)
        self.assertEqual(canonical(p), canonical(parse(print_program(p))))
        for change in ({'refresh': 'agent', 'l_max': 5, 'boot_out_level': 5}, {'placement': 'optimized'}):
            bad = replace(p, header=replace(p.header, **change))
            self.assertIn('暂不支持', str(typecheck(bad, {}).diags[0]))


class ManagementTests(unittest.TestCase):
    def test_five_unit_programs_check_and_roundtrip(self):
        for name in ('add', 'chain', 'residual', 'constants', 'no-final-mul'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                b, f = record_case(name)
                self.assertTrue(check(f))
                path = Path(temporary)/'managed.mlir'
                emit(f, path)
                parsed = read(path, f.profile)
                self.assertTrue(check(parsed))
                ids = {n.id: i for i, n in enumerate(f.nodes)}
                self.assertEqual([(n.kind, [ids.get(a, a) for a in n.args], n.type) for n in f.nodes],
                                 [(n.kind, n.args, n.type) for n in parsed.nodes])
                for a, z in zip(f.nodes, parsed.nodes):
                    if a.kind == 'constant':
                        np.testing.assert_array_equal(a.data, z.data)

    def test_cuts_use_the_single_depth_dp(self):
        for name in ('add', 'chain', 'residual', 'constants', 'no-final-mul'):
            b, f = record_case(name)
            from pt_agent.manage.greedy import placement_view
            p = f.profile
            table = {k.removeprefix('earth.'): v for k, v in p['latencyTable'].items()}
            result = depth_dp(placement_view(b.nodes, b._inputs, [v.nid for v in b._outputs]),
                last_boot_level=4, boot_in_level=1, boot_out_level=5, latency_table=table)
            self.assertEqual(f.cuts, result.cuts)
            self.assertEqual(sum(n.kind == 'bootstrap' for n in f.nodes), result.boots)

    def test_sequential_rms_matches_frontend(self):
        data = np.linspace(-1, 2, 1024)
        total = 0.
        for v in data:
            total += float(v)*float(v)
        self.assertEqual(rms(data, 'separate'), (total/len(data))**.5)

    def test_constant_encodings_are_private(self):
        b, f = record_case('constants')
        constants = [n for n in f.nodes if n.kind == 'constant']
        self.assertEqual(len(constants), 2)
        self.assertIs(constants[0].data, constants[1].data)
        self.assertNotEqual(constants[0].type, constants[1].type)

    def test_six_corruptions_have_operation_and_ptdsl_location(self):
        for kind in ('delete-rescale', 'downFactor', 'delete-bootstrap', 'result-level',
                     'bootstrap-scale', 'input-level'):
            with self.subTest(kind=kind):
                profile = toolchain.management_profile()
                # The same source-supported interface with factor != waterline
                # makes omission of refresh's upscale observable in a unit fixture.
                if kind == 'bootstrap-scale':
                    profile['rescalingFactor'] = 51
                _, f = record_case('chain', profile)
                f = deepcopy(f)
                if kind.startswith('delete-'):
                    target = 'rescale' if kind == 'delete-rescale' else 'bootstrap'
                    n = next(n for n in f.nodes if n.kind == target)
                    for other in f.nodes:
                        other.args = [n.args[0] if a == n.id else a for a in other.args]
                    f.outputs = [n.args[0] if a == n.id else a for a in f.outputs]
                    f.nodes.remove(n)
                elif kind == 'downFactor':
                    n = next(n for n in f.nodes if n.kind == 'modswitch')
                    n.attrs['downFactor'] += 1
                elif kind == 'result-level':
                    n = next(n for n in f.nodes if n.kind == 'mul')
                    n.type = replace(n.type, level=n.type.level+1)
                elif kind == 'bootstrap-scale':
                    n = next(n for n in f.nodes if n.kind == 'bootstrap')
                    parent = next(m for m in f.nodes if m.id == n.args[0])
                    # refineReturnValues may insert a modswitch after the upscale.
                    while parent.kind == 'modswitch':
                        parent = next(m for m in f.nodes if m.id == parent.args[0])
                    self.assertEqual(parent.kind, 'upscale')
                    n.args[0] = parent.args[0]
                else:
                    f.inputs[0] = replace(f.inputs[0], level=5)
                with self.assertRaises(ManagementError) as error:
                    check(f)
                self.assertIn('operation', str(error.exception))
                self.assertIn('PT-DSL op 12, line 7', str(error.exception))

    def test_source_capacity_rule_rejects_mul_lhs(self):
        p = toolchain.management_profile()
        from pt_agent.manage.ir import Type
        with self.assertRaisesRegex(ValueError, 'capacity'):
            infer('mul', [Type(True, 200, 3), Type(False, 43, 3)], {}, p)


@require(COMPILER)
class CommandsAndPackageTests(unittest.TestCase):
    def test_managed_arguments_derive_original_plan_options(self):
        original = toolchain.compile_argv('dacapo-depth-dp', 'qwen48', Path('.'))
        commands = toolchain.managed_compile_argv('dacapo-depth-dp', 'qwen48', Path('.'))
        self.assertEqual(commands[0][-3], '--pass-pipeline=builtin.module(canonicalize,cse,cse,canonicalize,'
                         'func.func(convert-earth-to-ckks,convert-upscale-to-mulcp,cse))')
        for arg in original:
            if arg.startswith('--runtime-plan-'):
                option = arg.removeprefix('--runtime-plan-')
                if option.startswith(('format=', 'id=')):
                    option = 'plan-'+option
                self.assertIn(option, commands[1][-3])
        substitutions = dict(hecate_opt='/hecate-opt', input='/out/managed.mlir',
                             package='/out', output_dir='/out')
        actual = [[a.format(**substitutions) for a in argv] for argv in managed_compile_template()]
        expected = [[a.replace('/out/boot-compiler-config.json', '/out/config/compiler-config.json')
                     for a in argv] for argv in commands]
        self.assertEqual(actual, expected)

    def test_managed_package_trace_without_hecate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = export_package(SMALL, parse(SMALL), {}, root)
            output = root/'managed'
            process = subprocess.run([sys.executable, '-B', str(package/'trace_model.py'), '--mode', 'managed',
                '--hecate-opt',str(toolchain.host_paths()['hecate_opt']),
                '--weights', str(root/'weights/square.npz'), '--output-dir', str(output)],
                capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertTrue((output/'managed.mlir').is_file())
            self.assertTrue(check(read(output/'managed.mlir', toolchain.management_profile())))
            meta = json.loads((package/'config/management.json').read_text())
            traced = json.loads((output/'management.json').read_text())
            self.assertEqual(meta['cuts'], traced['cuts'])
            self.assertEqual(meta['boots'], traced['boots'])
            self.assertEqual(meta['profile']['rms_accumulation'], 'fma')
            signature = json.loads((output/'signature.json').read_text())
            self.assertEqual(signature['management_emit']['rms_accumulation'], 'fma')
            self.assertTrue(all(n['op_id'] is not None and n['lineno'] > 0 for n in traced['values']))


if __name__ == '__main__':
    unittest.main()
