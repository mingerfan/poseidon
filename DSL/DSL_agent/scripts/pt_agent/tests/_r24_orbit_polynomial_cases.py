"""R24 synthetic float64 polynomial fixtures, with no saved models or samples."""

import ast
from contextlib import ExitStack
import copy
import inspect
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


MPCB_SOURCE = '''import r24_forbidden_top_level
raise AssertionError('MPCB top-level must not execute')
def fint(i):
    return int(np.floor(i))
def GenPoly(treeStr, coeffStr, length, scale=1.0):
    tree = [token.strip().split(" ") for token in treeStr]
    tree = [[int(istr) for istr in istrl] for istrl in tree]
    coeff = [float(token.strip())/scale for token in coeffStr]
    cheby_mish = np.polynomial.Chebyshev(np.array(coeff, dtype=np.double))
    newtree = [[num for num in arr] for arr in tree]
    newtree[0][0] = cheby_mish
    for i, cousins in enumerate(tree):
        for j, divisor in enumerate(cousins):
            if divisor > 0:
                divisor = np.polynomial.Chebyshev([0] * divisor + [1])
                newtree[i+1][2*j+1] = newtree[i][j] // divisor
                newtree[i+1][2*j] = newtree[i][j] % divisor
    calc_order = [(i,j,divisor,newtree[i][j]) for i,cousins in enumerate(tree)
                  for j,divisor in enumerate(cousins) if divisor >= 0]
    calc_order = sorted(calc_order, reverse=True, key=lambda x:x[0])
    def polynomial(x):
        giantTs = {0:1, 1:x}
        for i in range(1, fint(np.log2(length))):
            idx, pre_idx = pow(2,i), pow(2,i-1)
            giantTs[idx] = 2 * giantTs[pre_idx] * giantTs[pre_idx] + -1
        babyTs = [x]
        for i in range(1, fint(np.log2(length))):
            idx = pow(2,i)
            babyAdd = [2 * poly * giantTs[idx] for poly in babyTs]
            sdfs = [new+old for new,old in zip(babyAdd,reversed(babyTs))]
            babyAdd = [new-old for new,old in zip(babyAdd,reversed(babyTs))]
            babyTs = babyTs + babyAdd
        tmpPoly = {}
        for i,j,deg,leaf in calc_order:
            if deg == 0:
                poly = 0
                for k in range(length//2):
                    if len(leaf.coef) > 2*k+1:
                        poly += leaf.coef[2*k+1] * babyTs[k]
                tmpPoly[(i,j)] = poly
            else:
                if deg not in giantTs:
                    giantTs[deg] = 2 * giantTs[deg//2] * giantTs[deg//2] + -1
                tmpPoly[(i,j)] = tmpPoly[(i+1,2*j+1)] * giantTs[deg] + tmpPoly[(i+1,2*j)]
        return tmpPoly[(0,0)]
    return polynomial
'''
POLY_SOURCE = '''raise AssertionError('Poly top-level must not execute')
def GenPoly(degree=16):
    return MPCB.GenPoly(tree_var, coeff_var, degree)
'''
FUNC_SOURCE = '''raise AssertionError('Func top-level must not execute')
def HE_SiLU(name, x):
    calculation = Poly.GenPoly()
    hc.add_comment(f"// poly@{name}_SiLU_poly")
    y = calculation(x)
    hc.add_comment(f"// add@{name}_SiLU_add")
    y = y + 0.5
    hc.add_comment(f"// mul@{name}_SiLU_mul")
    y = x * y
    return y
hc.save('forbidden')
'''
TREE = '64\n16 16\n0 32 0 0\n-1 -1 16 0 -1 -1 -1 -1\n-1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n'


class PolynomialCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        from pt_agent.experiments import r24_orbit_polynomial_worker as worker
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        worker._initialize_torch()
        cls.np, cls.torch = worker._numeric_modules()
        cls.worker, cls.Recorder = worker, StageRecorder

    def setUp(self):
        temporary = TemporaryDirectory(prefix='r24-inner-', dir=os.environ.get('R24_TEST_WORK'))
        self.addCleanup(temporary.cleanup)
        self.case = Path(temporary.name)
        self.work = self.case / 'work/checks-worker'; self.work.mkdir(parents=True)
        self.recorder = self.Recorder(self.work / 'events.jsonl')
        self.poly = self.case / 'poly'; (self.poly/'data').mkdir(parents=True)
        for name, text in (('MPCB.py',MPCB_SOURCE),('Poly.py',POLY_SOURCE),('Func.py',FUNC_SOURCE)):
            (self.poly/name).write_text(text)
        (self.poly/'data/treeStr.txt').write_text(TREE)
        self.coefficients = self.np.zeros(96, dtype=self.np.float64)
        self.coefficients[[0,1,3,5,95]] = [0.5, 0.25, -0.03125, 0.0078125, -0.000001]
        (self.poly/'data/coeffStr.txt').write_text(''.join(str(float(value))+'\n' for value in self.coefficients))
        self.args=SimpleNamespace(work_dir=self.work,poly_dir=self.poly,
            model_source=self.case/'model.py',checkpoint=self.case/'weights.fixture',
            probes=self.case/'inputs.fixture',probe_source=self.case/'input-source.json',
            **{name+'_result':self.case/(name+'-result.json') for name in ('r13','r16','r22','r23')})

    def _model(self,*,short_forward=False):
        torch=self.torch
        class Block(torch.nn.Module):
            def __init__(self):super().__init__();self.mish=torch.nn.SiLU()
        class FixtureModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.gain=torch.nn.Parameter(torch.ones((),dtype=torch.float64))
                self.bn=torch.nn.BatchNorm1d(1);self.mish=torch.nn.SiLU()
                self.blocks=torch.nn.ModuleList([Block() for _ in range(9)])
                for index in range(110):self.register_buffer('fixture_'+str(index),torch.zeros((),dtype=torch.float64))
            def forward(self,images):
                value=images.mean(dim=(1,2,3)).reshape(-1,1)*self.gain
                value=self.mish(self.bn(value))
                for index,block in enumerate(self.blocks):
                    value=block.mish(value)
                    if not(short_forward and index==8):value=block.mish(value)
                return value.expand(-1,10).clone()
        return FixtureModel().double().eval()

    def _run(self,*,references=None,short_forward=False,reference_math=None):
        model=self._model(short_forward=short_forward)
        state={name:value.clone() for name,value in model.state_dict().items()}
        self.assertEqual(len(state),116)
        report=dict(passed=True,strict_load=True,extra_state=[],
            structure=dict(passed=True,activation_occurrences=19),
            original_state=[dict(name=name,finite=True,numerically_equal=True) for name in state])
        images=self.np.zeros((3,3,32,32),dtype=self.np.float64)
        labels=self.np.array([6,9,9],dtype=self.np.int64)
        references=references if references is not None else [self.np.zeros((1,10),dtype=self.np.float64) for _ in range(3)]
        source=dict(split='train',indices=[0,1,2],sample_ids=['train:0','train:1','train:2'],labels=labels.tolist(),
            normalization=dict(mean=[.485,.456,.406],std=[.229,.224,.225]))
        result=self.worker._initial_result();self.last_result=result;self.last_model=model
        with ExitStack() as stack:
            checked=stack.enter_context(patch.object(self.worker,'_check_sources',return_value=({'fixture':True},{},{})))
            loaded=stack.enter_context(patch.object(self.worker,'_load_state',return_value=state))
            built=stack.enter_context(patch.object(self.worker,'_build_model',return_value=(model,report)))
            probes=stack.enter_context(patch.object(self.worker,'_load_probes',return_value=(images,labels,references,source)))
            stack.enter_context(patch.object(self.worker.shutil,'disk_usage',return_value=SimpleNamespace(free=11*1024**3)))
            if reference_math is not None:
                stack.enter_context(patch.object(self.worker,'odd_chebyshev_reference',side_effect=reference_math))
            self.worker.run_checks(self.args,self.recorder,result)
        return result,dict(checked=checked,loaded=loaded,built=built,probes=probes),model

    def test_ast_loads_only_whitelisted_original_function_bodies(self):
        native, coefficients, metadata = self.worker.load_native_functions(self.poly)
        self.assertEqual(tuple(inspect.signature(native).parameters), ('name','x'))
        self.assertEqual(coefficients.shape, (96,))
        self.assertEqual(coefficients.dtype, self.np.dtype('float64'))
        self.np.testing.assert_array_equal(coefficients, self.coefficients)
        self.assertNotIn('r24_forbidden_top_level', sys.modules)
        json.dumps(metadata, allow_nan=False)
        values = self.np.array([[-0.5,-0.125],[0.0,0.375]], dtype=self.np.float64)
        comment=Mock();native.__globals__['hc'].add_comment=comment
        result = native('fixed_fixture', values)
        self.assertEqual([call.args[0] for call in comment.call_args_list],
            ['// poly@fixed_fixture_SiLU_poly','// add@fixed_fixture_SiLU_add','// mul@fixed_fixture_SiLU_mul'])
        self.assertEqual(native.__globals__['Poly'].GenPoly.__defaults__,(16,))
        self.assertEqual((metadata['poly_closure_attempts'],metadata['poly_closure_completed']),(1,1))
        self.assertEqual(metadata['closure_lengths'],[16]);self.assertEqual(metadata['closure_scales'],[1.0])
        reference = self.worker.odd_chebyshev_reference(coefficients, values)
        self.np.testing.assert_allclose(result, reference, atol=1e-12, rtol=1e-12)

    def test_ast_rejects_duplicate_missing_async_and_decorated_definitions(self):
        variants = [('MPCB.py',MPCB_SOURCE+'\ndef fint(i):\n    return i\n'),
                    ('MPCB.py',MPCB_SOURCE.replace('def GenPoly(', 'def unrelated(')),
                    ('Poly.py',POLY_SOURCE.replace('def GenPoly(', 'async def GenPoly(')),
                    ('Func.py',FUNC_SOURCE.replace('def HE_SiLU(', '@unknown\ndef HE_SiLU('))]
        for name, text in variants:
            path=self.poly/name; original=path.read_text(); path.write_text(text)
            try:
                with self.subTest(name=name), self.assertRaises((ValueError,RuntimeError)):
                    self.worker.load_native_functions(self.poly)
            finally:path.write_text(original)

    def test_odd_reference_removes_c0_and_even_terms_and_keeps_input_unchanged(self):
        coefficients = self.np.zeros(96, dtype=self.np.float64)
        coefficients[[0,1,2,3]] = [100.0,0.25,-200.0,0.0625]
        values=self.np.array([[-0.75,-0.125],[0.0,0.5]], dtype=self.np.float64)
        before_coefficients,before_values=coefficients.copy(),values.copy()
        reference=self.worker.odd_chebyshev_reference(coefficients,values)
        expected=values*(0.25*values+0.0625*(4*values*values*values-3*values)+0.5)
        self.np.testing.assert_allclose(reference,expected,atol=1e-12,rtol=1e-12)
        self.np.testing.assert_array_equal(coefficients,before_coefficients)
        self.np.testing.assert_array_equal(values,before_values)
        self.assertEqual(reference.dtype,self.np.dtype('float64'))
        self.assertEqual(reference.shape,values.shape)

    def test_original_tree_handles_nonmonomial_odd_terms_and_omits_even_part(self):
        first,coefficients,_=self.worker.load_native_functions(self.poly)
        values=self.np.array([-0.8,-0.2,0.0,0.3,0.9],dtype=self.np.float64)
        before=first('first',values)
        altered=coefficients.copy(); altered[0]=123.0; altered[2]=9.0
        (self.poly/'data/coeffStr.txt').write_text(''.join(str(float(v))+'\n' for v in altered))
        second,_,_=self.worker.load_native_functions(self.poly)
        self.np.testing.assert_allclose(second('second',values),before,atol=1e-12,rtol=1e-12)
        self.np.testing.assert_allclose(before,self.worker.odd_chebyshev_reference(coefficients,values),
                                        atol=1e-12,rtol=1e-12)

    def test_native_loader_rejects_wrong_coefficient_count_and_nonfinite_constants(self):
        path=self.poly/'data/coeffStr.txt';original=path.read_text()
        for text in ('0.5\n0.25\n',original.replace('0.5\n','nan\n',1)):
            path.write_text(text)
            with self.subTest(text=text[:30]),self.assertRaises(self.worker.WorkerFailure):
                self.worker.load_native_functions(self.poly)

    def test_math_check_keeps_original_tolerance_shape_dtype_and_finite_policy(self):
        reference=self.np.zeros((1,10),dtype=self.np.float64)
        self.assertTrue(self.worker._check_math(reference,reference.copy(),logits=True)['passed'])
        for observed in (self.np.ones((1,10),dtype=self.np.float64)*1e-8,
                         self.np.zeros((10,),dtype=self.np.float64),
                         self.np.zeros((1,10),dtype=self.np.float32),
                         self.np.full((1,10),self.np.nan,dtype=self.np.float64)):
            with self.subTest(shape=observed.shape,dtype=observed.dtype):
                self.assertFalse(self.worker._check_math(reference,observed,logits=True)['passed'])

    def test_physical_scale_is_applied_once_and_original_state_bn_are_preserved(self):
        original=self._model();approximate=copy.deepcopy(original)
        before={name:value.clone() for name,value in original.state_dict().items()}
        result=self.worker._initial_result();coefficients=self.np.zeros(96,dtype=self.np.float64)
        native=Mock(side_effect=lambda name,u:u*0.5)
        names=self.worker._replace_activations(approximate,native,coefficients,self.args,self.recorder,result)
        self.assertEqual(len(names),10)
        values=self.torch.tensor([[-64.0,16.0]],dtype=self.torch.float64)
        with self.torch.no_grad():output=approximate.mish(values)
        self.np.testing.assert_array_equal(native.call_args.args[1],self.np.array([[-2.0,.5]],dtype=self.np.float64))
        self.np.testing.assert_array_equal(output.numpy(),self.np.array([[-32.0,8.0]],dtype=self.np.float64))
        self.assertEqual(result['activation_attempts'],1);self.assertEqual(result['math_check_completed'],1)
        self.assertEqual(result['activation_rows'][0]['outside_standard_interval_elements'],1)
        for name,value in original.state_dict().items():
            self.assertTrue(self.torch.equal(value,before[name]))
            self.assertTrue(self.torch.equal(approximate.state_dict()[name],before[name]))
        self.assertEqual(original.bn.num_batches_tracked.dtype,self.torch.int64)
        self.assertTrue(self.worker._state_isolated(original,approximate)['passed'])

    def test_activation_rejects_non_double_input_before_any_native_arithmetic(self):
        approximate=copy.deepcopy(self._model());result=self.worker._initial_result();native=Mock()
        self.worker._replace_activations(approximate,native,self.coefficients,self.args,self.recorder,result)
        with self.assertRaises(self.worker.WorkerFailure):
            approximate.mish(self.torch.zeros((1,1),dtype=self.torch.float32))
        native.assert_not_called();self.assertEqual(result['activation_attempts'],1)
        self.assertEqual(result['activation_completed'],0);self.assertEqual(result['math_check_attempts'],0)

    def test_complete_synthetic_run_records_three_original_three_approx_and_fifty_seven_checks(self):
        result,hooks,model=self._run()
        for hook in hooks.values():hook.assert_called_once()
        for name in ('checkpoint_load','model_build','probe_load','deepcopy','native_load'):
            self.assertEqual((result[name+'_attempts'],result[name+'_completed']),(1,1))
        for name in ('original_forward','approx_forward'):
            self.assertEqual((result[name+'_attempts'],result[name+'_completed']),(3,3))
        for name in ('activation','math_check'):
            self.assertEqual((result[name+'_attempts'],result[name+'_completed']),(57,57))
        self.assertEqual(result['original_reference_checks'],3);self.assertEqual(result['silu_modules_replaced'],10)
        self.assertTrue(result['original_reference_passed']);self.assertTrue(result['native_polynomial_math_passed'])
        self.assertTrue(result['physical_scaled_forward_completed'])
        self.assertEqual(len(result['activation_rows']),57)
        self.assertEqual(result['native_metadata']['poly_closure_attempts'],57)
        self.assertEqual(result['native_metadata']['poly_closure_completed'],57)
        self.assertEqual(set(result['native_metadata']['closure_lengths']),{16})
        self.assertTrue(all(value==0 for value in (result[name] for name in self.worker.ZERO_FIELDS)))
        self.assertTrue(all(result[name] is None for name in self.worker.UNMEASURED_FIELDS))
        self.assertFalse(result['native_packed_equivalence_verified']);self.assertFalse(result['formal_matrix_complete'])
        self.assertEqual(sum(isinstance(module,self.torch.nn.SiLU) for module in model.modules()),10)

    def test_original_reference_failure_stops_before_clone_and_approximation(self):
        references=[self.np.zeros((1,10),dtype=self.np.float64) for _ in range(3)]
        references[1][0,0]=1.0
        with self.assertRaises(self.worker.WorkerFailure) as error:self._run(references=references)
        self.assertEqual(error.exception.status,'reference_failed')
        result=self.last_result;self.assertEqual(result['original_forward_completed'],2)
        self.assertEqual(result['original_reference_checks'],2);self.assertFalse(result['original_reference_passed'])
        self.assertEqual(result['deepcopy_attempts'],0);self.assertEqual(result['approx_forward_attempts'],0)

    def test_local_math_failure_retains_successful_original_references_and_partial_counts(self):
        def wrong_reference(coefficients,values):return self.np.ones_like(values)
        with self.assertRaises(self.worker.WorkerFailure) as error:self._run(reference_math=wrong_reference)
        self.assertEqual(error.exception.status,'math_failed')
        result=self.last_result;self.assertTrue(result['original_reference_passed'])
        self.assertFalse(result['native_polynomial_math_passed']);self.assertFalse(result['physical_scaled_forward_completed'])
        self.assertEqual((result['approx_forward_attempts'],result['approx_forward_completed']),(1,0))
        self.assertEqual((result['activation_attempts'],result['activation_completed']),(1,1))
        self.assertEqual((result['math_check_attempts'],result['math_check_completed']),(1,1))

    def test_eighteen_activation_forward_is_rejected_without_retry(self):
        with self.assertRaises(self.worker.WorkerFailure) as error:self._run(short_forward=True)
        self.assertEqual(error.exception.status,'structure_failed')
        result=self.last_result;self.assertTrue(result['original_reference_passed'])
        self.assertEqual(result['approx_forward_attempts'],1);self.assertEqual(result['approx_forward_completed'],1)
        self.assertEqual(result['activation_completed'],18);self.assertEqual(result['math_check_completed'],18)
        self.assertFalse(result['physical_scaled_forward_completed'])


if __name__ == '__main__':
    unittest.main()
