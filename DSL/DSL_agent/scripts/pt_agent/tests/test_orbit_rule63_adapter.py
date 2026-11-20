"""Small same-program format/level/cost fixtures; no model or solver runs."""
import importlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.tests._artifacts import require
from pt_agent.experiments import orbit_rule63_adapter as a


def profile():
    keys = ['add_single', 'add_double', 'mul_single', 'mul_double', 'rotate_single',
        'negate_single', 'modswitch_single', 'rescale_single', 'bootstrap_single']
    return dict(polynomialDegree=65536, bootstrapLevelLowerBound=1,
        bootstrapLevelUpperBound=5, levelLowerBound=1, levelUpperBound=8,
        rescalingFactor=43, runtime='HEAAN-HEVM',
        latencyTable={'earth.' + k: [0] + [100 * (j + 1) + l for l in range(1, 9)]
            for j, k in enumerate(keys)})


def raw(body):
    return ('"builtin.module"() ({\n"func.func"() ({\n'
        '^bb0(%arg0: tensor<1x!earth.ci<0 * 0>> loc(unknown)):\n' + body +
        '\n}) : () -> ()\n}) : () -> ()\n')


class AdapterTest(unittest.TestCase):
    def test_dense_conversion_ids_payload_and_operation_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            payload = struct.pack('<2d', -0.0, 1.25)
            source = p / 'raw.mlir'
            source.write_text(raw('%0 = "earth.constant"() <{rms_var = 1.25 : f64, value = dense<"0x' +
                payload.hex() + '"> : tensor<2xf64>}> : () -> tensor<1x!earth.pl<0 * 0>>\n'
                '%1 = "earth.add"(%arg0, %0) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>\n'
                '"func.return"(%1) : (tensor<1x!earth.ci<0 * 0>>) -> ()'))
            inventory = a.adapt_mlir(source, p / 'adapted.mlir', p / 'constants.cst', p / 'ledger.json')
            self.assertEqual(inventory['ops_original'], {'constant': 1, 'add': 1})
            self.assertEqual(inventory['ops_adapted'], inventory['ops_original'])
            self.assertEqual(inventory['records'][0]['ssa'], '%0')
            self.assertIn('rms_var = 1.25 : f64, value = 0 : i64', (p / 'adapted.mlir').read_text())
            # Synthetic conversion output is inspected; no scientific input readback.
            self.assertEqual((p / 'constants.cst').read_bytes(), struct.pack('<qq', 1, 2) + payload)

    def test_decimal_splat_and_poly_types_rejection(self):
        self.assertEqual(a.dense_payload('1.5', 2), struct.pack('<2d', 1.5, 1.5))
        self.assertEqual(a.dense_payload('[[-1.0, 2.0]]', 2), struct.pack('<2d', -1, 2))
        self.assertEqual(a.map_types('tensor<1x!ckks.poly<2 * 43 * 8>> tensor<1x!ckks.poly<1 * 43 * 8>>'),
            'tensor<1x!earth.ci<43 * 8>> tensor<1x!earth.pl<43 * 8>>')
        with self.assertRaises(ValueError): a.dense_payload('[1,2]', 1)
        with self.assertRaises(ValueError): a.map_types('!ckks.unknown<1>')

    def test_params_physical_levels_and_cost_indices(self):
        class Params:
            def __init__(self, *args): pass
            def check_res(self, il, isc, ol, osc):
                return il >= ol and 43 * il - isc >= 43 * ol - osc
        p = a.make_params(profile(), Params)
        self.assertEqual((p.poly_deg, p.max_slot, p.bts_lb, p.bts_ub, p.lvl_ub), (65536,16384,1,5,8))
        self.assertFalse(p.dacapo_mlir_in)
        self.assertTrue(p.check_resbts(1,43,5,43))
        self.assertFalse(p.check_resbts(1,43,6,43))
        table = a.adapted_cost_config(profile())['latencyTable']
        self.assertEqual(table['earth.mul_single'][0], profile()['latencyTable']['earth.mul_single'][1])
        self.assertEqual(table['earth.upscale_single'][4], profile()['latencyTable']['earth.mul_single'][5])
        self.assertEqual(set(table['earth.bootstrap_single']), {profile()['latencyTable']['earth.bootstrap_single'][1]})
        bad = profile(); bad['polynomialDegree'] = 32768
        with self.assertRaises(ValueError): a.validate_profile(bad)

    def test_fixed_constraint_and_context_restoration(self):
        calls = []
        class Expr:
            def __le__(self, x): return ('<=',x)
        class Model:
            def addGenConstrIndicator(self, *args, **kw): calls.append((args,kw))
        vp = SimpleNamespace(model=Model(), var_use=lambda *x:'boot',var_lvl=lambda *x:Expr())
        dag = SimpleNamespace(nodes={'a':{'op':'negate'},'c':{'op':'constant'}})
        original = lambda *args:calls.append('original')
        edge = object()
        ilp, decoder = SimpleNamespace(add_ilp_constraints=original),SimpleNamespace(_decode_edge=edge)
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with a.fixed_boot_hooks(ilp,decoder):
                ilp.add_ilp_constraints(dag,vp)
                self.assertIs(decoder._decode_edge,a.fixed_decode_edge)
                raise RuntimeError('fixture')
        self.assertEqual(calls[0],'original')
        self.assertEqual(calls[1][0],('boot',True,('<=',5)))
        self.assertIs(ilp.add_ilp_constraints, original)
        self.assertIs(decoder._decode_edge, edge)


@require(paths.ORBIT_ROOT / "scripts/tdag/tdag.py", paths.ORBIT_ROOT / "scripts/params/params.py")
class OfficialTinyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        # Explicit public library checkout solely for synthetic fixtures.
        root = paths.ORBIT_ROOT
        if str(root) not in sys.path: sys.path.insert(0,str(root))
        from scripts.params.params import Params
        from scripts.tdag.tdag import Tdag
        cls.Params, cls.Tdag = Params, Tdag

    def params(self): return a.make_params(profile(), self.Params)

    def test_fixed_decoder_actual_insertions_weighted_prediction_and_serialized_calls(self):
        p = self.params(); g = self.Tdag(p,'fixture')
        g.add_node('u',op='negate',weight=1,level=8,scale=86,op_descr={},comment='')
        g.add_node('v',op='negate',weight=1,level=4,scale=43,op_descr={},comment='')
        g.add_edge('u','v',weight=1)
        # 8/86 ->4/43 needs no bootstrap; input-level rescale pricing remains8.
        a.fixed_decode_edge(g,'u','v',None)
        self.assertEqual(sum(g.nodes[n]['op']=='bootstrap' for n in g.nodes),0)
        g = self.Tdag(p,'fixtureboot')
        g.add_node('u',op='input',weight=1,level=2,scale=43,op_descr={},comment='')
        g.add_node('v',op='negate',weight=1,level=4,scale=43,op_descr={},comment='')
        g.add_edge('u','v',weight=1)
        a.fixed_decode_edge(g,'u','v',None)
        boots = [n for n in g.nodes if g.nodes[n]['op']=='bootstrap']
        self.assertEqual(len(boots),1)
        b = boots[0]; bp = next(g.predecessors(b))
        self.assertEqual((g.nodes[bp]['level'],g.nodes[b]['level']),(1,5))
        self.assertEqual(sum(g.nodes[n]['op']=='modswitch' for n in g.nodes),2)
        # Input u is not an operation. v is the real negate after the reductions.
        o = a.decoded_observation(g,profile())
        self.assertEqual(o['compiled_bootstraps'],1)
        self.assertEqual(o['execution_ops']['modswitch'],2)
        self.assertEqual(o['pred_latency_s'],o['pred_latency_without_upscale_s'])
        from scripts.tdag.tdag2mlir import tdag_to_mlir
        g.inputs={'u'};g.outputs={'v'}
        with tempfile.TemporaryDirectory() as t:
            target=Path(t)/'out.mlir';tdag_to_mlir(g,str(target))
            self.assertEqual(a.serialized_bootstrap_calls(target)['calls'],1)
        g.nodes[b]['level']=6
        with self.assertRaises(ValueError): a.decoded_observation(g,profile())

    def test_estimator_fixed_bootprice_and_upscale_delta(self):
        from scripts.latency_estimator.latency_estimator import LatencyEstimator
        p = self.params(); e = a.make_estimator(p,profile(),LatencyEstimator)
        self.assertEqual(e.lin_op_lmaps['bootstrap_single'],(0,901))
        self.assertEqual(e.resbts_cost(1,43,4,43),901+705)
        g=self.Tdag(p,'scale')
        g.add_node('u',op='input',weight=1,level=5,scale=43,op_descr={},comment='')
        g.add_node('s',op='upscale',weight=1,level=5,scale=86,op_descr={'upFactor':43},comment='')
        g.add_edge('u','s',weight=1)
        o=a.decoded_observation(g,profile())
        self.assertEqual(o['pred_latency_s'],305/1e6)
        self.assertEqual(o['pred_latency_without_upscale_s'],0)
        bad=profile();del bad['latencyTable']['earth.mul_single']
        o=a.decoded_observation(g,bad)
        self.assertIsNone(o['pred_latency_s']);self.assertEqual(len(o['missing']),1)

    def test_full_pipeline_tiny_no_solver(self):
        import sys
        from scripts.assignment.assignment import Assign
        partition=importlib.import_module('scripts.optimizer.orbit.iterative_partition')
        def solve(comp,manager,prev,e,p):
            self.assertEqual(prev,{8:{43:0}})
            assignment=Assign(comp)
            inp=next(iter(comp.inputs));out=next(iter(comp.outputs))
            assignment.v_lvl_in[inp]=8;assignment.v_scl_in[inp]=43
            assignment.v_lvl_out[inp]=1;assignment.v_scl_out[inp]=43
            assignment.v_lvl_out[out]=5;assignment.v_scl_out[out]=43
            for u,v in comp.edges:
                assignment.e_lvl_out[(u,v)]=1;assignment.e_scl_out[(u,v)]=43
            return {(8,43):{(5,43):assignment}},{(8,43):{(5,43):999}}
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);source=p/'raw.mlir'
            source.write_text(raw('%0 = "earth.negate"(%arg0) : (tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>\n"func.return"(%0) : (tensor<1x!earth.ci<0 * 0>>) -> ()'))
            with patch.object(partition,'solve_partition',solve):
                result=a.compile_rule63(source_mlir=source,profile=profile(),work_dir=p/'work',
                    orbit_root=paths.ORBIT_ROOT)
            self.assertEqual(result['compiled_bootstraps'],1)
            self.assertEqual(result['serialized_bootstrap_inventory']['calls'],1)
            self.assertEqual(result['status'],'ok')
            self.assertEqual(result['input_inventory']['ops'],{'negate':1})
            self.assertTrue(result['compile_verified'])
            self.assertTrue(result['bootstrap_weighted_matches_serialized'])
            self.assertIn('init_level = 8 : i64', (p/'work/optimized-hecate-depth.mlir').read_text())
            self.assertIn('targetLevel = 3 : i64', (p/'work/optimized-hecate-depth.mlir').read_text())

    def test_fixed_input_mult_partition_result_from_original_manager(self):
        marker = object()
        class Manager:
            pdag_name_to_qbp = {'tiny': marker}
            def get_qbp_assign(self, graph):
                return {(8,43): {(5,43): marker}}
            def get_qbp_cost(self, name):
                return {(8,43): {(5,43): 4}}
        partition = SimpleNamespace(solve_partition=lambda *args: None)
        solved = a.solve_fixed_input(partition, Manager(), SimpleNamespace(name='tiny'), None, None)
        self.assertIs(solved[0][(8,43)][(5,43)], marker)
        self.assertEqual(solved[1][(8,43)][(5,43)],4)


if __name__ == '__main__': unittest.main()
