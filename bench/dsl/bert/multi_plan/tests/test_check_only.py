#!/usr/bin/env python3
"""Permanent CPU gate tests. Synthetic tiny schema assets, no ignored goldens.

Run: python3 .../test_check_only.py --runner /path/bert_multi_plan_runner
     --output-dir test-results/bert-a800-20261115/multi-plan-build/tests
Optional --guard /path/no_device_guard.so proves no CUDA device entry points.
These are verifier/compatibility mocks, NOT claims of GPU-valid HE parameters.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ARGS = None
RESULTS = []

def sha(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, separators=(',', ':')) + '\n')

def assets(root):
    host = dict(kind='host', rank=0)
    dev = dict(kind='device', rank=0, index=0)
    q = [12289, 18433, 40961]
    profile = dict(profile_id='mock-native', input_level_min=0, input_level_max=2,
                   output_level=2, output_scale_log2=28, implementation='native',
                   input_components=2, output_components=2,
                   latency_us_by_input_level=[1]*3, noise_by_input_level=None,
                   host_requirements=dict(needs_secret_key=False, needs_host_compute=False))
    spec = dict(spec_format_version=2, spec_id='mock-spec', version=1,
                status='validated', target_id='poseidon-ckks-gpu', rescale_mode='native',
                noise_unit=None,
                provenance=dict(kind='research-manual-schedule', repository='test',
                                revision='mock', path='mock', source_sha256=sha(b'mock')),
                context=dict(context_id='mock', poly_degree=512, rns_moduli_log2=[14,15,16],
                             rns_moduli=q, max_modulus_log2=16, default_scale_log2=14),
                levels=dict(lower_bound=0, upper_bound=2),
                operators=dict(rescale=dict(supported=True, latency_us_by_level=[1]*3,
                                            noise_by_level=None, max_levels_per_op=3),
                               boot=dict(supported=True)), boot_profiles=[profile])
    for op in ['add_cc','add_cp','sub_cc','sub_cp','mul_cc','mul_cp','negate','rotate','mod_switch','relinearize']:
        spec['operators'][op]=dict(supported=True,latency_us_by_level=[1]*3,noise_by_level=None)
    dump(root/'spec.json', spec)
    values = []
    def desc(ident, place, level, scale):
        return dict(id=str(ident), kind='ciphertext', place=place, context='mock',
                    level=level, scale_log2=scale, ntt=True, components=2)
    for i in range(3):
        values += [desc(2*i, host, 2, 28), desc(2*i+1, dev, 2, 28)]
    for i in range(3):
        values.append(desc(6+i, dev, 0, 10))
        values.append(desc(9+i, dev, 2, 28))
    init = [dict(kind='transfer', ordinal=i, transfer_id=str(i), hint='auto',
                 inputs=[str(2*i)], outputs=[str(2*i+1)], sources=[host],
                 destinations=[dev], output_kinds=['ciphertext']) for i in range(3)]
    execution = [dict(kind='compute', op='ApiCall', ordinal=3+i, inputs=['1','3','5'],
                      output=str(6+i), place=dev,
                      attrs=dict(callee='bert.compact_linear', call_id=str(i))) for i in range(3)]
    execution += [dict(kind='compute', op='boot', ordinal=6+i, inputs=[str(6+i)],
                       output=str(9+i), place=dev,
                       attrs=dict(target_level=2, target_scale_log2=28, target_components=2,
                                  operator_profile='mock-native', implementation='native')) for i in range(3)]
    plan = dict(format_version=7, plan_id='1',
                target=dict(target_id='poseidon-ckks-gpu', world_size=1, device_counts=[1],
                            capability_version=1, operator_spec=dict(id='mock-spec', version=1,
                                                                    source_sha256=sha((root/'spec.json').read_bytes()))),
                values=values, external_inputs=['0','2','4'], initialization=init,
                execution=execution, finalization=[], final_outputs=['9','10','11'])
    weights = b'\0'*(768*768*4)
    bias = b'\0'*(768*4)
    (root/'weights.bin').write_bytes(weights+bias)
    rotations = list(range(1,16)) + [16*g for g in range(-16,16) if g]
    matrix = dict(matrix_id='layer0.q_weight', input_width=768, output_width=768,
                  weight_offset=0, bias_offset=len(weights), tile=256, baby_step=16)
    calls = [dict(call_id=str(i), matrix_id='layer0.q_weight', output_tile=i,
                  input_width=768, output_width=768, tile=256, baby_step=16,
                  input_level=2, input_scale_log2=28, output_level=0,
                  output_scale_log2=10, rescale_count=2) for i in range(3)]
    manifest = dict(format_version=1, dtype='<f4', weight_file='weights.bin',
                    weight_sha256=sha(weights+bias), matrices=[matrix], calls=calls,
                    rotation_steps=sorted(rotations))
    dump(root/'linear.json', manifest)
    fields = ['q_weight','q_bias','k_weight','k_bias','v_weight','v_bias','o_weight','o_bias',
              'ffn_up_weight','ffn_up_bias','ffn_down_weight','ffn_down_bias',
              'attention_ln_gamma','attention_ln_beta','ffn_ln_gamma','ffn_ln_beta']
    parameters = {key:sha(b'unsupplied-mock-field:'+key.encode()) for key in fields}
    parameters.update(q_weight=sha(weights), q_bias=sha(bias))
    rows = [[float(group+1)/8]*256 for group in range(3)]
    original = b''.join(struct.pack('<d', x) for row in rows for x in row)
    fixture = dict(format_version=1,
                   context=dict(poly_degree=512, q_moduli_bits=[14,15,16], p_moduli_bits=[17],
                                q_moduli=q, p_moduli=[65537], default_scale_log2=14,
                                secret_hamming_weight=128, q0_level=1),
                   native_boot=dict(profile_id='mock-native'), compact_linear_manifest='linear.json',
                   compact_periodic_encoding=False, prepared_rotation_steps=sorted(rotations),
                   prepared_direct_rotation_keys=True,
                   inputs=[dict(value_id=2*i, values=row) for i,row in enumerate(rows)],
                   expected_outputs=[dict(value_id=9+i, values=row) for i,row in enumerate(rows)],
                   model=dict(layout='cerium-feature-tiled', seed=20261114, config=dict(tokens=1, tile=256,
                              hidden=768, heads=3, slots=256, intermediate=768, layers=1,
                              bsgs_baby_step=16, reciprocal_iterations=3, ln_epsilon=1e-5,
                              ln_variance_center=1.0, weight_std=0.02, attention_layout='offset'),
                              input_sha256=sha(original), weight_sha256={'0':parameters}))
    for label in ['a','b']:
        dump(root/(label+'.plan.json'), plan)
        dump(root/(label+'.fixture.json'), fixture)
    schedule = dict(plans=[dict(name=x.upper(), plan=x+'.plan.json', spec='spec.json',
                               fixture=x+'.fixture.json') for x in ['a','b']],
                    schedule=[dict(name=x, phase='diagnostic') for x in ['A','A','B','A']])
    dump(root/'schedule.json', schedule)

class CheckOnly(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mock-', dir=ARGS.output_dir)
        self.root = Path(self.temp.name)
        assets(self.root)
    def tearDown(self):
        self.temp.cleanup()
    def mutate(self, name, fn):
        path = self.root/name
        value = json.loads(path.read_text())
        fn(value)
        dump(path, value)
    def run_check(self, expected=None):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        if ARGS.guard:
            env['LD_PRELOAD'] = str(ARGS.guard)
        proc = subprocess.run([str(ARGS.runner), '--check-only', str(self.root/'schedule.json'),
                               str(self.root/'report.json')], env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        report = json.loads((self.root/'report.json').read_text())
        if ARGS.guard:
            self.assertIn('BERT_NO_DEVICE_GUARD loaded device_calls=0',proc.stderr)
            self.assertIn('BERT_NO_DEVICE_GUARD finished device_calls=0',proc.stderr)
            self.assertNotIn('blocked api=',proc.stderr)
        RESULTS.append(dict(test=self.id(),exit=proc.returncode,guard=bool(ARGS.guard),
                            device_calls_zero=('finished device_calls=0' in proc.stderr),
                            report_completed=report['completed'],error=report.get('error')))
        if expected is None:
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(report['completed'])
            self.assertTrue(report['compatibility']['compatible'])
            self.assertFalse(report['gpu_executed'])
            self.assertEqual(report['keys']['generation_calls'], 0)
            self.assertEqual(report['native_boot_profile']['generation_calls'], 0)
        else:
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertFalse(report['completed'])
            self.assertIn(expected, report['error'])
        return report
    def test_same_material_aaba(self):
        self.run_check()
    def test_model_ln_epsilon_only(self):
        self.mutate('b.fixture.json',lambda f:f['model']['config'].__setitem__('ln_epsilon',2e-5))
        self.run_check('model_config')
    def test_model_variance_center_only(self):
        self.mutate('b.fixture.json',lambda f:f['model']['config'].__setitem__('ln_variance_center',1.5))
        self.run_check('model_config')
    def test_model_reciprocal_iterations_only(self):
        self.mutate('b.fixture.json',lambda f:f['model']['config'].__setitem__('reciprocal_iterations',4))
        self.run_check('model config requires exactly three reciprocal iterations')
    def test_model_layers_only(self):
        self.mutate('b.fixture.json',lambda f:f['model']['config'].__setitem__('layers',2))
        self.run_check('parameter hash layer set must equal model config layers')
    def test_model_layers_requires_positive_integer(self):
        for value in [0,-1,1.5,True,'1']:
            with self.subTest(layers=value):
                self.mutate('b.fixture.json',lambda f:f['model']['config'].__setitem__('layers',value))
                self.run_check('layers must be a positive integer')
    def test_parameter_hash_wrong_layer_set(self):
        self.mutate('b.fixture.json',lambda f:f['model']['weight_sha256'].__setitem__('1',f['model']['weight_sha256'].pop('0')))
        self.run_check('parameter hash layer set must be canonical 0..layers-1')
    def test_parameter_hash_noncanonical_layer_set(self):
        self.mutate('b.fixture.json',lambda f:f['model']['weight_sha256'].__setitem__('00',f['model']['weight_sha256'].pop('0')))
        self.run_check('parameter hash layer set must be canonical 0..layers-1')
    def test_parameter_hash_extra_layer(self):
        self.mutate('b.fixture.json',lambda f:f['model']['weight_sha256'].__setitem__('1',copy.deepcopy(f['model']['weight_sha256']['0'])))
        self.run_check('parameter hash layer set must equal model config layers')
    def test_model_heads_dimensions(self):
        self.mutate('b.fixture.json',lambda f:f['model']['config'].__setitem__('heads',5))
        self.run_check('invalid model config dimensions')
    def test_model_explicit_head_dim_mismatch(self):
        self.mutate('b.fixture.json',lambda f:f['model']['config'].__setitem__('head_dim',64))
        self.run_check('head_dim disagrees with hidden/heads')
    def test_model_intermediate_only(self):
        self.mutate('b.fixture.json',lambda f:f['model']['config'].__setitem__('intermediate',1536))
        self.run_check('model_config')
    def test_model_missing_semantic_field(self):
        self.mutate('b.fixture.json',lambda f:f['model']['config'].pop('ln_epsilon'))
        self.run_check('model config missing required field ln_epsilon')
    def test_model_numeric_semantic_type(self):
        self.mutate('b.fixture.json',lambda f:f['model']['config'].__setitem__('ln_epsilon',True))
        self.run_check('ln_epsilon must be finite and positive')
    def test_model_top_level_experiment_flags_may_differ(self):
        self.mutate('b.fixture.json',lambda f:f['model'].update(prepared_qk=True,prepared_av=True,algebra_rewrite={'enabled':True}))
        self.run_check()
    def test_changed_input_with_stale_hash(self):
        self.mutate('b.fixture.json', lambda f:f['inputs'][0]['values'].__setitem__(0,0.333))
        self.run_check('actual original client input SHA256')
    def test_different_valid_input_hash(self):
        def change(f):
            f['inputs'][0]['values'][0]=0.333
            f['model']['input_sha256']=sha(b''.join(struct.pack('<d',x) for r in f['inputs'] for x in r['values']))
        self.mutate('b.fixture.json',change)
        self.run_check('original_input')
    def test_changed_context_p(self):
        self.mutate('b.fixture.json', lambda f:f['context'].__setitem__('p_moduli',[114689]))
        self.run_check('context')
    def test_fixture_spec_context_mismatch(self):
        self.mutate('b.fixture.json', lambda f:f['context'].__setitem__('default_scale_log2',15))
        self.run_check('fixture/spec default scale')
    def test_changed_ln_hash(self):
        self.mutate('b.fixture.json', lambda f:f['model']['weight_sha256']['0'].__setitem__('ffn_ln_gamma',sha(b'other')))
        self.run_check('parameters')
    def test_missing_parameter_hash(self):
        self.mutate('b.fixture.json', lambda f:f['model']['weight_sha256']['0'].pop('ffn_ln_beta'))
        self.run_check('all 16 parameter')
    def test_corrupt_weights(self):
        path=self.root/'weights.bin'
        raw=bytearray(path.read_bytes());raw[0]=1;path.write_bytes(raw)
        self.run_check('actual Linear weight file SHA256')
    def test_manifest_namespace_duplicate(self):
        self.mutate('linear.json', lambda m:m['calls'].append(copy.deepcopy(m['calls'][0])))
        self.run_check('duplicate callee/call_id namespace')
    def test_same_namespace_different_contract(self):
        m=json.loads((self.root/'linear.json').read_text())
        for call in m['calls']:call['output_scale_log2']=9
        dump(self.root/'other-linear.json',m)
        self.mutate('b.fixture.json',lambda f:f.__setitem__('compact_linear_manifest','other-linear.json'))
        self.run_check('namespace collision')
    def test_plan_descriptor_manifest_mismatch(self):
        self.mutate('a.plan.json',lambda p:next(d for d in p['values'] if d['id']=='6').__setitem__('scale_log2',9))
        self.run_check('physical descriptor mismatch')
    def test_linear_sequence_operand_ids(self):
        self.mutate('a.plan.json',lambda p:p['execution'][1].__setitem__('inputs',['3','1','5']))
        self.run_check('exact operand ids changed')
    def test_shared_rescale_configuration(self):
        spec=json.loads((self.root/'spec.json').read_text());spec['rescale_mode']='lazy'
        dump(self.root/'other-spec.json',spec)
        self.mutate('b.plan.json',lambda p:p['target']['operator_spec'].__setitem__('source_sha256',sha((self.root/'other-spec.json').read_bytes())))
        self.mutate('schedule.json',lambda s:s['plans'][1].__setitem__('spec','other-spec.json'))
        self.run_check('rescale_mode')
    def test_shared_max_rescale_configuration(self):
        spec=json.loads((self.root/'spec.json').read_text());spec['operators']['rescale']['max_levels_per_op']=2
        dump(self.root/'other-spec.json',spec)
        self.mutate('b.plan.json',lambda p:p['target']['operator_spec'].__setitem__('source_sha256',sha((self.root/'other-spec.json').read_bytes())))
        self.mutate('schedule.json',lambda s:s['plans'][1].__setitem__('spec','other-spec.json'))
        self.run_check('operators')
    def test_three_plan_library(self):
        def third(s):
            entry=copy.deepcopy(s['plans'][0]);entry['name']='C';s['plans'].append(entry)
            s['schedule'].append(dict(name='C',phase='warmup'))
        self.mutate('schedule.json',third)
        self.run_check()
    def large_context(self):
        # Still a schema mock: enough genuine N65536 primes to test CPU attention validators.
        q=[204865537,205651969,206307329,207880193,209059841,210370561,
           211025921,211812353,214171649,215482369,215744513,216137729,216924161]
        spec=json.loads((self.root/'spec.json').read_text())
        spec['context'].update(poly_degree=65536,rns_moduli_log2=[28]*len(q),rns_moduli=q,
                               max_modulus_log2=28,default_scale_log2=28)
        spec['levels']['upper_bound']=len(q)-1
        spec['boot_profiles'][0].update(input_level_max=len(q)-1,latency_us_by_input_level=[1]*len(q))
        for op in spec['operators'].values():
            if 'latency_us_by_level' in op:op['latency_us_by_level']=[1]*len(q)
        dump(self.root/'spec.json',spec)
        for label in ['a','b']:
            self.mutate(label+'.plan.json',lambda p:p['target']['operator_spec'].__setitem__('source_sha256',sha((self.root/'spec.json').read_bytes())))
            def change(f):
                f['context'].update(poly_degree=65536,q_moduli=q,q_moduli_bits=[28]*len(q),
                                    p_moduli=[268042241],p_moduli_bits=[28],default_scale_log2=28)
                f['model']['config'].update(tokens=128,slots=32768,heads=12,attention_layout='cyclic')
                for row in f['inputs']+f['expected_outputs']:row['values']*=128
                raw=b''.join(struct.pack('<d',f['inputs'][g]['values'][t*256+x])
                             for t in range(128) for g in range(3) for x in range(256))
                f['model']['input_sha256']=sha(raw)
            self.mutate(label+'.fixture.json',change)
        return q
    def test_av_headroom_rejected_before_gpu_setup(self):
        import math
        q=self.large_context()
        drop=lambda level:math.log2(q[level])+math.log2(q[level-1])
        score=65.0;value=2*score-drop(10);common=value+value-drop(8);output=2*common-drop(6)
        call=dict(call_id='0',id='mock-av',tokens=128,tile=256,head_dim=64,baby=8,
                  score_input_level=10,score_input_scale_log2=score,value_input_level=8,
                  value_input_scale_log2=value,common_level=6,common_scale_log2=common,
                  output_level=4,output_scale_log2=output)
        dump(self.root/'av.json',dict(format_version=1,calls=[call],rotation_steps=[]))
        def bind(f):
            f['prepared_av_manifest']='av.json';f['prepared_av_manifest_sha256']=sha((self.root/'av.json').read_bytes())
        self.mutate('a.fixture.json',bind)
        self.run_check('capacity/headroom')
    def test_qk_internal_contract_field_required(self):
        self.large_context()
        call=dict(call_id='0',id='mock-qk',tokens=128,tile=256,head_dim=64,baby=8,
                  input_level=8,input_scale_log2=55,output_level=4,output_scale_log2=55)
        dump(self.root/'qk.json',dict(format_version=1,calls=[call],rotation_steps=[]))
        def bind(f):
            f['prepared_qk_manifest']='qk.json';f['prepared_qk_manifest_sha256']=sha((self.root/'qk.json').read_bytes())
        self.mutate('a.fixture.json',bind)
        self.run_check('intermediate_scale_log2')
    def test_unknown_callee(self):
        self.mutate('a.plan.json',lambda p:p['execution'][0]['attrs'].__setitem__('callee','other.namespace'))
        self.run_check('unbound prepared callee/call_id')
    def test_waterline_fail_closed(self):
        self.mutate('b.fixture.json',lambda f:f.__setitem__('waterline',{}))
        self.run_check('waterline is not supported')
    def test_prepared_rotation_manifest_audit(self):
        self.mutate('linear.json',lambda m:m['rotation_steps'].pop())
        self.run_check('rotation steps disagree')
    def test_unknown_schedule_plan(self):
        self.mutate('schedule.json',lambda s:s['schedule'][2].__setitem__('name','unknown'))
        self.run_check('unknown scheduled plan')
    def test_standard_plan_verifier_not_bypassed(self):
        self.mutate('a.plan.json',lambda p:p['execution'][3]['attrs'].__setitem__('operator_profile','nonexistent'))
        self.run_check('plan verification failed')
    def test_noargs_and_help(self):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        if ARGS.guard:env['LD_PRELOAD']=str(ARGS.guard)
        for args in [[],['--help']]:
            p=subprocess.run([str(ARGS.runner)]+args,env=env,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertIn('usage:',p.stdout)
            if ARGS.guard:
                self.assertIn('BERT_NO_DEVICE_GUARD loaded device_calls=0',p.stderr)
                self.assertIn('BERT_NO_DEVICE_GUARD finished device_calls=0',p.stderr)
                self.assertNotIn('blocked api=',p.stderr)
            RESULTS.append(dict(test=self.id(),args=args,exit=p.returncode,guard=bool(ARGS.guard),
                                device_calls_zero=('finished device_calls=0' in p.stderr)))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--runner',required=True,type=Path)
    parser.add_argument('--guard',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('test-results/bert-a800-20261115/multi-plan-build/attention-modes/tests-37'))
    ARGS,remaining=parser.parse_known_args()
    ARGS.runner=ARGS.runner.resolve()
    if ARGS.guard:ARGS.guard=ARGS.guard.resolve()
    ARGS.output_dir.mkdir(parents=True,exist_ok=True)
    program=unittest.main(argv=[__file__]+remaining,exit=False)
    dump(ARGS.output_dir/'no-device-gate.json',dict(passed=program.result.wasSuccessful(),
                                                 tests_run=program.result.testsRun,invocations=RESULTS))
    raise SystemExit(0 if program.result.wasSuccessful() else 1)
