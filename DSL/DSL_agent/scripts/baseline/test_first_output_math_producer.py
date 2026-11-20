"""Mathematical producer controls, with synthetic graphs and inert native metadata.

No generated fixture source is an Agent output. Nothing here compiles, encrypts,
decrypts or executes a native backend or original uploaded model.
"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from benchmark_suite import Builder
from component_contract import prepare_task
from delivery_component import bound, canonical, read_bytes, sha
from first_output_assessment import compare, evaluate_witness
from first_output_dsl_math import evaluate, MathematicalWorkLimit
import first_output_math_producer as producer


def square_request():
    builder = Builder([(2,)])
    return prepare_task(builder.finish(builder.node('square', ['input0'])))


class DSLMathematicalTests(unittest.TestCase):
    def setUp(self):
        self.request = square_request()
        self.source = b'@hc.func("c,c")\r\ndef golden(x, zero_ct):\r\n    return x*x\r\n\r\n'
        self.inputs = [[.5, -.25, 0., 0.], [0., 0., 0., 0.]]

    def run_math(self, source=None, request=None, inputs=None, budget=2**28):
        return evaluate(self.source if source is None else source,
            self.request if request is None else request,
            self.inputs if inputs is None else inputs, maximum_scalar_operations=budget)

    def test_actual_source_math_not_a_backend_or_candidate_python_execution(self):
        result = self.run_math()
        self.assertEqual(result['values'], [.25, .0625])
        self.assertFalse(result['candidate_python_executed'])
        self.assertFalse(result['encrypted_execution'])

    def test_wrong_legal_DSL_is_observed_without_repair_or_reference_substitution(self):
        result = self.run_math(self.source.replace(b'x*x', b'x'))
        self.assertEqual(result['values'], [.5, -.25])
        self.assertNotEqual(result['values'], [.25, .0625])

    def test_rotation_direction_follows_independent_slot_definition(self):
        b = Builder([(4,)])
        request = prepare_task(b.finish(b.node('rotate', ['input0'], step=1)))
        source = b'@hc.func("c,c")\ndef golden(x, zero_ct):\n    return x.rotate(1)\n'
        result = self.run_math(source, request, [[.1, .2, .3, .4], [0., 0., 0., 0.]])
        self.assertEqual(result['values'], [.2, .3, .4, .1])

    def test_helpers_preserve_typed_call_boundaries(self):
        source = b'@hc.func("c")\ndef squared(a):\n    return a*a\n@hc.func("c,c")\ndef golden(x, zero_ct):\n    return squared(x)\n'
        self.assertEqual(self.run_math(source)['values'], [.25, .0625])

    def test_array_alias_mutation_and_copy_are_distinct(self):
        b = Builder([(2,)])
        request = prepare_task(b.finish(b.node('add', ['input0', 'input0']), 'input0'))
        source = b'@hc.func("c,c")\ndef golden(x, zero_ct):\n    a=np.array([x,x],dtype=object)\n    alias=a\n    saved=a.copy()\n    a+=x\n    return [alias[0],saved[0]]\n'
        self.assertEqual(self.run_math(source, request)['values'], [1., -.5, .5, -.25])

    def test_public_loop_and_scalar_augmentation_do_not_mutate_input_alias(self):
        b = Builder([(2,)])
        request = prepare_task(b.finish(b.node('add', ['input0', 'input0'])))
        source = b'@hc.func("c,c")\ndef golden(x, zero_ct):\n    y=zero_ct\n    for i in range(2):\n        y+=x\n    return y\n'
        self.assertEqual(self.run_math(source, request)['values'], [1., -.5])

    def test_arbitrary_code_and_recursive_calls_are_rejected_before_execution(self):
        for source in [b'@hc.func("c,c")\ndef golden(x,zero_ct):\n    __import__("os").system("false")\n    return x\n',
                b'@hc.func("c,c")\ndef golden(x,zero_ct):\n    return golden(x,zero_ct)\n']:
            with self.assertRaises(ValueError): self.run_math(source)

    def test_finite_work_budget_is_enforced(self):
        with self.assertRaises(MathematicalWorkLimit): self.run_math(budget=1)

    def test_nonfinite_mathematical_result_is_not_accepted(self):
        with self.assertRaises((ValueError, FloatingPointError)):
            self.run_math(inputs=[[1e308, 1., 0., 0.], [0., 0., 0., 0.]])

    def test_physical_zero_argument_and_output_shape_cannot_be_guessed(self):
        with self.assertRaises(ValueError): self.run_math(inputs=[self.inputs[0]])
        with self.assertRaises(ValueError): self.run_math(inputs=[[.5, -.25], self.inputs[1]])


class MathematicalProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only discovery is cached. produce_record still hashes every frozen
        # source before and after each calculation.
        cls.bundle = producer.implementation_bundle()
        cls.sources = cls.bundle['sources']

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.request = square_request()
        self.original = self.request['model']
        self.samples = [dict(input0=[.5, -.25]), dict(input0=[.25, .125]), dict(input0=[-.125, .375])]
        self.source = b'@hc.func("c,c")\r\ndef golden(x, zero_ct):\r\n    return x*x\r\n\r\n'
        def cached(extra_sources=()):
            result = copy.deepcopy(self.bundle)
            result['sources'].update({str(Path(p)): sha(read_bytes(p)) for p in extra_sources})
            return result
        self.cache = mock.patch.object(producer, 'implementation_bundle', side_effect=cached)
        self.cache.start(); self.addCleanup(self.cache.stop)

    def policy(self, **kwargs):
        options = dict(domain={i['name']: [-1., 1.] for i in self.original['inputs']},
            approximation_requirements=[], application_requirements=[])
        options.update(kwargs)
        return producer.make_policy(self.request, self.original, self.samples, **options)

    def record(self, policy=None, **kwargs):
        selection = producer.produce_record(self.request, self.source, self.original, self.samples,
            self.policy() if policy is None else policy, self.root/'math.json', **kwargs)
        import json
        return json.loads(read_bytes(selection['path'])), selection

    def test_source_closure_covers_model_reference_AST_and_independent_arithmetic(self):
        names = {Path(p).name for p in self.sources}
        self.assertTrue({'benchmark_math.py', 'operator_reference.py', 'decorated_functions.py',
            'first_output_dsl_math.py', 'first_output_math_producer.py'} <= names)
        self.assertGreaterEqual(len(set(self.sources.values())), 5)

    def test_three_input_groups_preserve_exact_source_hash_and_readonly_record(self):
        record, descriptor = self.record()
        self.assertEqual(record['source_sha256'], sha(self.source))
        self.assertEqual(len(record['inputs']), 3)
        self.assertEqual(record['inputs'][0]['original_reference'], [.25, .0625])
        self.assertEqual(record['inputs'][0]['declared_reference'], [.25, .0625])
        self.assertEqual(record['inputs'][0]['dsl_mathematical_output'], [.25, .0625])
        self.assertFalse(record['encrypted_execution'])
        self.assertFalse(record['CKKS_outputs_used_as_reference'])
        self.assertEqual(Path(descriptor['path']).stat().st_mode & 0o777, 0o400)

    def test_legal_wrong_DSL_has_nonzero_math_error_and_is_not_repaired(self):
        self.source = self.source.replace(b'x*x', b'x')
        record, _ = self.record()
        row = record['inputs'][0]
        result = compare(row['declared_reference'], row['dsl_mathematical_output'], atol=1e-12, rtol=1e-12)
        self.assertEqual(result['failed_values'], 2)
        self.assertEqual(record['source_sha256'], sha(self.source))

    def test_original_function_declared_approximation_and_DSL_stay_separate(self):
        from model_decomposition import FORMAT, decompose
        self.original = dict(format=FORMAT, id='producer_relu_fixture', inputs=[dict(name='x', shape=[2])],
            constants={}, nodes=[dict(id='activation', op='relu', inputs=['x'], attrs=dict(
                approximation=dict(method='polynomial', basis='power', domain=[-1., 1.], coefficients=[0., .5, .5])),
                outputs=['y'])], outputs=[dict(name='output0', value='y')])
        graph, _ = decompose(self.original)
        self.request = prepare_task(graph)
        self.samples = [dict(x=[-.5, .25])]
        self.source = b'@hc.func("c,c")\ndef golden(x,zero_ct):\n    return x*.5+(x*x)*.5\n'
        record, _ = self.record()
        row = record['inputs'][0]
        self.assertEqual(row['original_reference'], [0., .25])
        self.assertEqual(row['declared_reference'], [-.125, .15625])
        self.assertEqual(row['dsl_mathematical_output'], [-.125, .15625])

    def test_input_domain_rejection_does_not_become_an_acceptance_claim(self):
        policy = self.policy(domain=dict(input0=[-.1, .1]))
        record, _ = self.record(policy)
        self.assertFalse(record['inputs'][0]['inside_frozen_domain'])

    def test_changed_inputs_and_original_model_identity_are_rejected(self):
        policy = self.policy(); self.samples[0]['input0'][0] = .75
        with self.assertRaises(ValueError): self.record(policy)
        self.samples[0]['input0'][0] = .5
        self.original = copy.deepcopy(self.original); self.original['id'] = 'changed'
        with self.assertRaises(ValueError): self.record(policy)

    def test_changed_frozen_producer_source_is_rejected(self):
        extra = self.root/'frozen_fixture_role.py'; extra.write_text('# independent fixture role\n')
        policy = self.policy(extra_sources=[extra]); extra.write_text('# changed\n')
        with self.assertRaises(ValueError): self.record(policy)

    def test_missing_native_original_reference_is_not_replaced_by_float64_graph(self):
        native = dict(manifest_sha256='0'*64, input_dtypes=dict(input0='float32'),
            output_structure={}, source_identity={})
        policy = self.policy(reference_kind='native_source', native_source=native)
        with self.assertRaises(ValueError): self.record(policy)
        self.assertFalse((self.root/'math.json').exists())

    def test_application_dataset_requirement_cannot_be_inferred_from_three_inputs(self):
        requirements = [dict(id='top1_drop', comparison='<=', limit=.01)]
        with self.assertRaises(ValueError): self.policy(application_requirements=requirements)
        policy = self.policy(application_requirements=requirements,
            application_contract=dict(samples=10000, dataset_sha256='a'*64,
                producer_sources={next(iter(self.sources)): next(iter(self.sources.values()))}))
        with self.assertRaises(ValueError): self.record(policy)

    def test_shortened_application_dataset_is_rejected(self):
        contract = dict(samples=10000, dataset_sha256='a'*64,
            producer_sources={next(iter(self.sources)): next(iter(self.sources.values()))})
        policy = self.policy(application_requirements=[dict(id='top1_drop', comparison='<=', limit=.01)],
            application_contract=contract)
        report = bound(dict(format='poseidon-first-output-application-evidence-v1', contract_sha256=sha(canonical(contract)),
            original_model_sha256=policy['original_model_sha256'], declared_model_sha256=sha(canonical(self.request['model'])),
            source_sha256=sha(self.source), samples=9999, dataset_sha256=contract['dataset_sha256'],
            producer_sources=contract['producer_sources'], candidate_python_executed=False,
            CKKS_outputs_used_as_reference=False, metrics=dict(top1_drop=0.)))
        p = self.root/'short-dataset-fixture.json'; p.write_bytes(canonical(report))
        with self.assertRaises(ValueError): self.record(policy, application_evidence=dict(path=str(p), sha256=sha(read_bytes(p))))

    def test_retained_reference_cannot_replace_a_different_original_function(self):
        original = copy.deepcopy(self.original); original['id'] = 'different_original'
        with self.assertRaises(ValueError):
            producer.make_policy(self.request, original, self.samples, domain=dict(input0=[-1., 1.]),
                approximation_requirements=[], application_requirements=[], reference_kind='retained_core', retained_reference={})

    def test_same_record_path_cannot_select_a_later_result(self):
        record, descriptor = self.record(); original_bytes = read_bytes(descriptor['path'])
        with self.assertRaises(FileExistsError): self.record()
        self.assertEqual(read_bytes(descriptor['path']), original_bytes)

    def test_mathematical_record_alone_never_supplies_compiler_or_CKKS_success(self):
        policy = self.policy(); record, _ = self.record(policy)
        result = evaluate_witness(dict(source_sha256=sha(self.source), test_only=True,
            compiled_validated=False, real_compiler=False, inputs=record['inputs']), policy, sha(self.source))
        self.assertFalse(result['raw_compile_passed'])
        self.assertFalse(result['raw_numeric_passed'])
        self.assertFalse(result['raw_joint_passed'])


if __name__ == '__main__': unittest.main()
