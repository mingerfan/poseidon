"""CPU-only checks for the manual schedule, never CKKS accuracy claims."""
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from backend import Graph, ManualLowering, coefficient_moduli


CONTEXT = dict(poly_degree=64, q_moduli_bits=[31] + [30]*31,
               p_moduli_bits=[31]*16, default_scale_log2=30)
PROFILE = dict(profile_id='bert-native30', input_level_min=2,
               input_level_max=31, output_level=17, output_scale_log2=30)


def replay(directory, graph, inputs):
    plan = json.loads((directory/'model.runtime-plan.json').read_text())
    desc = {v['id']: v for v in plan['values']}
    spec = json.loads((directory/'operator-spec.json').read_text())
    q = spec['context']['rns_moduli']
    constants = graph.constants
    values = {ident: inputs[name] for ident, name in zip(plan['external_inputs'], inputs)}
    for row in plan['initialization'] + plan['execution']:
        kind = row['kind']
        if kind == 'release': del values[row['value']]; continue
        if kind == 'encode':
            a = constants[row['payload']['content']]
            value = np.zeros(graph.slots); value[:len(a)] = a
        elif kind == 'transfer':
            for src, dst in zip(row['inputs'], row['outputs']): values[dst] = values[src]
            continue
        elif kind == 'compute':
            args = [values[i] for i in row['inputs']]; op = row['op']
            if op in ('add_cc', 'add_cp'): value = args[0] + args[1]
            elif op in ('sub_cc', 'sub_cp'): value = args[0] - args[1]
            elif op in ('mul_cc', 'mul_cp'): value = args[0] * args[1]
            elif op == 'negate': value = -args[0]
            elif op == 'rotate': value = np.roll(args[0], -row['attrs']['steps'])
            elif op in ('rescale', 'mod_switch', 'relinearize', 'boot'): value = args[0]
            else: raise AssertionError(op)
            out = desc[row['output']]
            ins = [desc[i] for i in row['inputs']]
            if op in ('add_cc', 'sub_cc', 'add_cp', 'sub_cp'):
                assert abs(ins[0]['scale_log2'] - ins[1]['scale_log2']) < 1e-9
                assert ins[0]['level'] == ins[1]['level'] == out['level']
            if op in ('mul_cc', 'mul_cp'):
                assert abs(out['scale_log2'] - sum(v['scale_log2'] for v in ins)) < 1e-9
            if op == 'rescale':
                actual = ins[0]['scale_log2'] - sum(math.log2(p) for p in q[out['level']+1:ins[0]['level']+1])
                assert abs(actual-out['scale_log2']) < 1e-9, (actual, out)
            if op == 'boot': assert row['attrs']['implementation'] == 'native'
        else: raise AssertionError(kind)
        values[row['output']] = value
    return [values[i] for i in plan['final_outputs']]


class ManualScheduleTests(unittest.TestCase):
    def test_residual_alignment_and_fused_sum(self):
        for resident in (True, False):
            g = Graph(32); x = g.input('x')
            a = x * g.plain(np.linspace(.1, .3, 32))
            b = x.rotate(3) * g.plain(np.linspace(.2, -.1, 32))
            y = (a+b).rotate(-2)
            y = y*y + x
            inputs = dict(x=np.arange(32)/100)
            with tempfile.TemporaryDirectory() as root:
                out = Path(root)/'plan'
                stats = ManualLowering(g, CONTEXT, PROFILE).export([y], out, resident=resident)
                np.testing.assert_allclose(replay(out, g, inputs), g.evaluate([y], inputs), atol=1e-12)
                self.assertEqual(stats['instruction_counts']['rescale'], 3)
                self.assertEqual(stats['instruction_counts']['boot'] if 'boot' in stats['instruction_counts'] else 0, 0)

    def test_budget_and_stage_native_boots(self):
        for strategy in ('budget', 'stage'):
            g = Graph(32); x = g.input('x'); y = x
            for i in range(20):
                y = y * y
                if i == 5: y = g.bootstrap(y, label='explicit-boundary')
            inputs = dict(x=np.full(32, .999999))
            with tempfile.TemporaryDirectory() as root:
                out = Path(root)/'plan'
                stats = ManualLowering(g, CONTEXT, PROFILE, strategy=strategy).export([y+x], out)
                np.testing.assert_allclose(replay(out, g, inputs), g.evaluate([y+x], inputs), atol=1e-10)
                self.assertGreater(stats['instruction_counts']['boot'], 0)
                self.assertEqual('explicit-boundary' in stats['boot_labels'], strategy == 'stage')

    def test_binary_mask_folding_preserves_bsgs_reduction(self):
        from packed_model import PackedConfig, linear
        config=PackedConfig(tokens=4,hidden=6,intermediate=11,heads=3,slots=32,row_stride=16)
        g=Graph(32); inputs=[g.input('x'+str(i)) for i in range(2)]
        mask=g.plain(np.tile(np.r_[np.ones(6),np.zeros(10)],2))
        weight=np.arange(66).reshape(11,6)/100+.01
        outputs=[linear(g,x*mask,weight,np.zeros(11),config=config) for x in inputs]
        data={x.data:np.arange(32)/100 for x in inputs}
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'plan'
            stats=ManualLowering(g,CONTEXT,PROFILE).export(outputs,out)
            self.assertEqual(stats['instruction_counts']['rescale'],4)
            np.testing.assert_allclose(replay(out,g,data),g.evaluate(outputs,data),atol=1e-12)

    def test_complete_packed_encoder_lowering_and_release(self):
        from packed_model import PackedConfig, create_weights, build_model, pack_inputs, unpack_outputs
        from test_packed_model import dense_model
        config=PackedConfig(tokens=4,hidden=6,intermediate=11,heads=3,slots=32,row_stride=16)
        weights=create_weights(config,12)
        hidden=np.random.default_rng(13).standard_normal((4,6))
        hidden/=np.sqrt(np.mean(hidden*hidden,axis=1,keepdims=True))
        slots=pack_inputs(hidden,config,tail_value=.37)
        for strategy in ('stage','budget','selective'):
            for resident,fuse in ((True,True),(False,False)):
                g=Graph(32); inputs=[g.input('x'+str(i)) for i in range(2)]
                outputs=build_model(g,inputs,weights,config,boot_policy='stage')
                data={x.data:row for x,row in zip(inputs,slots)}
                with tempfile.TemporaryDirectory() as root:
                    out=Path(root)/'plan'
                    stats=ManualLowering(g,CONTEXT,PROFILE,strategy=strategy,fuse=fuse).export(outputs,out,resident=resident)
                    decoded=unpack_outputs(np.asarray(replay(out,g,data)),config)
                    np.testing.assert_allclose(decoded,dense_model(hidden,weights,config),atol=1e-12)
                    self.assertGreater(stats['memory_lower_bounds']['peak_payload_bytes']['device_ciphertext'],0)

    def test_shallow_linear_initial_level(self):
        g=Graph(32); x=g.input('x'); y=x*g.plain(np.arange(32)/100)
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'plan'
            stats=ManualLowering(g,CONTEXT,PROFILE,initial_level=5).export([y],out)
            self.assertEqual(stats['initial_level'],5)
            self.assertEqual(stats['instruction_counts'].get('boot',0),0)
            plan=json.loads((out/'model.runtime-plan.json').read_text())
            values={v['id']:v for v in plan['values']}
            self.assertEqual(values[plan['final_outputs'][0]]['level'],4)

    def test_prime_order_and_canonical_scales(self):
        primes = coefficient_moduli(65536, [30]*4)
        self.assertEqual(primes, [1068236801, 1070727169, 1071513601, 1073479681])
        scheduler = ManualLowering(Graph(32), CONTEXT, PROFILE)
        for level in range(1, len(scheduler.q)):
            self.assertAlmostEqual(2*scheduler.scales[level]-math.log2(scheduler.q[level]),
                                   scheduler.scales[level-1], places=11)


if __name__ == '__main__': unittest.main()
