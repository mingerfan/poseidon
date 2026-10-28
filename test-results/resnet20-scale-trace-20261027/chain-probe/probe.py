from pathlib import Path
from collections import Counter
import copy
import hashlib
import json
import math
import random
import re
import subprocess
import time

ROOT = Path('/home/xuming/poseidon-manual-boot-20261026')
AUDIT = ROOT/'resnet20-optimized-20261027T055506Z/placement-audit'
OUT = AUDIT/'chain-waste-probe'
OUT.mkdir(exist_ok=True)
SOURCE = AUDIT/'one-relu-tagged.mlir'
BASE = json.loads((AUDIT/'one-relu-full-spec.json').read_text())
COMPILER = ROOT/'hecate-opt'
GENERATOR = ROOT/'src/third_party/ckks-runtime/integrations/dacapo/generate_model_artifacts.py'
Q = BASE['context']['rns_moduli']


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n-1, 0
    while not d % 2:
        d //= 2
        s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n-1):
            continue
        for _ in range(s-1):
            x = x*x % n
            if x == n-1:
                break
        else:
            return False
    return True


used = set(Q)


def primes(bits, count):
    result = []
    stride = 2*BASE['context']['poly_degree']
    n = ((1 << bits)-2)//stride*stride+1
    while len(result) < count:
        if n not in used and is_prime(n):
            assert n.bit_length() == bits and n % stride == 1
            result.append(n)
            used.add(n)
        n -= stride
    return result


variants = {'manual': (Q, 30, 'Baseline actual manual primes')}
working = Q[6:31]
variants['reverse-working'] = (Q[:6]+list(reversed(working))+Q[31:], 30, 'Same primes, same Q count and exact total modulus; reverse only working-prime order')
for seed in (0, 1, 2):
    shuffled = working[:]
    random.Random(seed).shuffle(shuffled)
    variants[f'shuffle-{seed}'] = (Q[:6]+shuffled+Q[31:], 30, 'Same primes and exact total modulus; shuffle only working-prime order')
variants['working-all-30'] = (Q[:6]+primes(30,25)+Q[31:], 30, '25 working primes of 30 bits, 6 fewer nominal working bits than baseline')
variants['working-all-40'] = (Q[:6]+primes(40,25)+Q[31:], 30, 'Diagnostic only: 25 working primes of 40 bits; MORE total modulus bits')
new_work = primes(36,1)+primes(40,18)
variants['working-same-bits-40'] = (Q[:6]+new_work+Q[25:], 24, 'Diagnostic only: 19 working primes, nominal sum 756 bits equals baseline; actual total differs slightly')
target_log = sum(math.log2(q) for q in working)-sum(math.log2(q) for q in new_work[1:])
stride = 2*BASE['context']['poly_degree']
matched = (int(2**target_log)-1)//stride*stride+1
while matched in used or not is_prime(matched):
    matched -= stride
assert matched.bit_length() == 36
matched_work = [matched]+new_work[1:]
assert 0 <= sum(math.log2(q) for q in working)-sum(math.log2(q) for q in matched_work) < 1e-4
variants['working-matched-budget-40'] = (Q[:6]+matched_work+Q[31:], 24, 'Same full and working modulus bit budget (new total no larger, difference below 0.0001 bit); original 6 base and 19 Boot-internal primes retained')
variants['reverse-only-30'] = (Q[:9]+list(reversed(Q[9:31]))+Q[31:], 30, 'Same actual primes AND bit-width order; reverse only the 22 equal-width working primes')
for seed in (0,1):
    shuffled = Q[9:31]
    random.Random(seed).shuffle(shuffled)
    variants[f'shuffle-only-30-{seed}'] = (Q[:9]+shuffled+Q[31:], 30, 'Same actual primes AND bit-width order; shuffle only equal-width working primes')


def extract(plan, ir, chain, anchor):
    desc = {v['id']: v for v in plan['values']}
    producers = {}
    scalar = {}
    encoded_scalars = {}
    for line in ir.splitlines():
        if '"ckks.encode"' not in line:
            continue
        oid = re.search(r'(?:runtime.value_id|dist.logical_id) = (\d+)',line)[1]
        encoded_scalars[oid] = float(re.search(r'payload = dense<([-+0-9.eE]+)>',line)[1])
    kinds = Counter()
    mul_details = []
    prime_divisions = 0
    rescale_bits = 0.
    for op in plan['initialization']+plan['execution']+plan['finalization']:
        kind = op.get('op',op['kind'])
        outputs = op.get('outputs',[op['output']] if 'output' in op else [])
        if not outputs:
            continue
        oid = outputs[0]
        producers[oid] = op
        args = op.get('inputs',[])
        kinds[kind] += 1
        if kind == 'encode':
            scalar[oid] = encoded_scalars[oid]
        elif kind == 'transfer':
            scalar[oid] = scalar.get(args[0])
        elif kind.startswith('mul'):
            ps = [desc[a] for a in args]
            assert math.isclose(sum(v['scale_log2'] for v in ps),desc[oid]['scale_log2'],abs_tol=1e-8)
            value = scalar.get(args[1]) if kind == 'mul_cp' else None
            mul_details.append(dict(id=oid,kind=kind,unit=value==1,plaintext=value,level=desc[oid]['level'],input_scales=[v['scale_log2'] for v in ps],scale=desc[oid]['scale_log2']))
        elif kind == 'rescale':
            before, after = desc[args[0]],desc[oid]
            expected = before['scale_log2']-sum(math.log2(chain[l]) for l in range(before['level'],after['level'],-1))
            assert math.isclose(expected,after['scale_log2'],abs_tol=1e-8)
            prime_divisions += before['level']-after['level']
            rescale_bits += sum(math.log2(chain[l]) for l in range(before['level'],after['level'],-1))
        elif kind in ('add_cc','add_cp'):
            assert math.isclose(desc[args[0]]['scale_log2'],desc[args[1]]['scale_log2'],abs_tol=1e-8)
    roots = ['178','223','297']
    stage_outputs = {}
    for line in ir.splitlines():
        match = re.match(r'\s*%\w+ = "ckks\.(\w+)"',line)
        if not match:
            continue
        identifier = re.search(r'(?:runtime.value_id|dist.logical_id) = (\d+)',line)
        if not identifier:
            continue
        for i,root in enumerate(roots):
            if '"raw-'+root+'"' in line and match[1] == 'addcc':
                stage_outputs[i] = identifier[1]
    assert len(stage_outputs) == 3,stage_outputs
    final = plan['final_outputs'][0]
    outputs = [stage_outputs[i] for i in range(3)]+[final]
    consumed = [anchor-desc[oid]['level'] for oid in outputs]
    stages = []
    previous = 0
    for oid,depth in zip(outputs,consumed):
        stages.append(dict(id=oid,q=desc[oid]['level']+1,scale=desc[oid]['scale_log2'],consumed_q=depth-previous,total_consumed_q=depth))
        previous = depth
    units = [m for m in mul_details if m['unit']]
    # Intended plaintext replay: verifies graph semantics, NOT CKKS numerical precision.
    replay = []
    for x in (-.75,-.25,0.,.25,.75):
        values = {oid:x for oid in plan['external_inputs']}
        for op in plan['initialization']+plan['execution']+plan['finalization']:
            kind = op.get('op',op['kind'])
            out = op.get('outputs',[op['output']] if 'output' in op else [])
            if not out:
                continue
            oid = out[0]
            if kind == 'encode':
                assert scalar[oid] is not None
                values[oid] = scalar[oid]
                continue
            args = [values[a] for a in op.get('inputs',[])]
            if kind in ('mul_cc','mul_cp'):
                values[oid] = args[0]*args[1]
            elif kind in ('add_cc','add_cp'):
                values[oid] = args[0]+args[1]
            elif kind == 'negate':
                values[oid] = -args[0]
            else:
                assert kind in ('transfer','mod_switch','rescale','relinearize'),kind
                values[oid] = args[0]
        replay.append(dict(input=x,output=values[final]))
    output_level = desc[final]['level']
    output_capacity = sum(math.log2(q) for q in chain[:output_level+1])
    return dict(counts=dict(kinds),total_q_consumed=consumed[-1],stages=stages,unit_multiplications=len(units),unit_plain_scale_sum=sum(m['input_scales'][1] for m in units),rescale_prime_divisions=prime_divisions,rescale_total_log2=rescale_bits,output_modulus_log2=output_capacity,output_modulus_minus_scale=output_capacity-desc[final]['scale_log2'],multiplications=mul_details,plaintext_replay=replay)


results = []
for name,(chain,anchor,note) in variants.items():
    case = OUT/name
    case.mkdir(exist_ok=True)
    spec = copy.deepcopy(BASE)
    spec['context']['rns_moduli'] = chain
    spec['context']['rns_moduli_log2'] = [q.bit_length() for q in chain]
    spec['context']['max_modulus_log2'] = max(q.bit_length() for q in chain)
    spec['levels']['upper_bound'] = len(chain)-1
    for operator in spec['operators'].values():
        for key in ('latency_us_by_level','noise_by_level'):
            if isinstance(operator.get(key),list):
                operator[key] = operator[key][:len(chain)]
    spec['boot_profiles'][0]['latency_us_by_input_level'] = spec['boot_profiles'][0]['latency_us_by_input_level'][:len(chain)]
    spec['boot_profiles'][0]['output_level'] = anchor
    path = case/'operator-spec.json'
    path.write_text(json.dumps(spec,indent=2)+'\n')
    command = ['python3',str(GENERATOR),'mlp','--operator-spec',str(path),'--compiler-profile',str(ROOT/'alignment-tests/compiler.json'),'--hecate-opt',str(COMPILER),'--traced-mlir',str(SOURCE),'--output-dir',str(case/'artifacts'),'--device-counts','1','--plan-format','json','--boot-placement','greedy','--waterline','40']
    t0 = time.monotonic()
    proc = subprocess.run(command,capture_output=True,text=True)
    (case/'compile.log').write_text(proc.stdout+proc.stderr)
    row = dict(name=name,note=note,compile_exit_code=proc.returncode,compile_seconds=time.monotonic()-t0,input_q=anchor+1,working_prime_bits=sum(q.bit_length() for q in chain[6:anchor+1]),working_actual_log2=sum(math.log2(q) for q in chain[6:anchor+1]),active_modulus_log2=sum(math.log2(q) for q in chain[:anchor+1]),full_modulus_log2=sum(math.log2(q) for q in chain),moduli=chain)
    if proc.returncode == 0:
        plan_path = next((case/'artifacts').glob('*.runtime-plan.json'))
        plan = json.loads(plan_path.read_text())
        row.update(extract(plan,(case/'artifacts/mlp.optimized.mlir').read_text(),chain,anchor))
        verify = subprocess.run([str(ROOT/'validate-plan'),str(plan_path),str(path)],capture_output=True,text=True)
        row['validator_exit_code'] = verify.returncode
        row['validator_output'] = verify.stdout+verify.stderr
        assert verify.returncode == 0,row['validator_output']
        if name in ('manual','working-matched-budget-40'):
            align_spec = copy.deepcopy(spec)
            align_spec['boot_profiles'][0]['input_scale_log2'] = 40
            align_path = case/'alignment-spec.json'
            align_path.write_text(json.dumps(align_spec,indent=2)+'\n')
            final = row['stages'][-1]
            ct = f'tensor<1x!ckks.poly<2 * {math.ceil(final["scale"])} * {final["q"]-1}>>'
            result_type = f'tensor<1x!ckks.poly<2 * 40 * {anchor}>>'
            source = f'module {{ func.func @align(%x: {ct}) -> {result_type} attributes {{init_level = {anchor} : i64, earth.arg_scales = array<f64: {final["scale"]:.17e}>}} {{ %y = "ckks.bootstrapc"(%x) : ({ct}) -> {result_type}\n return %y : {result_type} }} }}'
            (case/'align.mlir').write_text(source)
            pipeline = f'builtin.module(func.func(materialize-ckks-physical-levels{{operator-spec={align_path} boot-profile=one-relu-full-trace-diagnostic levels-per-logical-level=1}}))'
            align = subprocess.run([str(COMPILER),f'--ckks-config={ROOT}/alignment-tests/compiler.json',f'--pass-pipeline={pipeline}',str(case/'align.mlir'),'-o',str(case/'aligned.mlir')],capture_output=True,text=True)
            (case/'alignment.log').write_text(align.stdout+align.stderr)
            row['boot_input_alignment'] = dict(exit_code=align.returncode,accepted=align.returncode==0)
            if align.returncode == 0:
                text = (case/'aligned.mlir').read_text()
                row['boot_input_alignment'].update(rescales=text.count('"ckks.rescalec"'),mod_switches=text.count('"ckks.modswitchc"'))
                assert re.search(r'!ckks.poly<2 \* (40|41) \* 5>',text)
            else:
                assert name == 'manual' and 'insufficient Q/encoding precision' in align.stderr,align.stderr
            assert row['boot_input_alignment']['accepted'] == (name != 'manual')
    else:
        row['error'] = (proc.stdout+proc.stderr)[-3500:]
    results.append(row)
    (OUT/'results.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),waterline=40,results=results),indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k not in ('moduli','multiplications','validator_output')},ensure_ascii=False),flush=True)

baseline = results[0]
for row in results[1:]:
    if row['compile_exit_code'] == 0:
        for left,right in zip(baseline['plaintext_replay'],row['plaintext_replay']):
            assert math.isclose(left['output'],right['output'],rel_tol=1e-12,abs_tol=1e-12),(row['name'],left,right)
print('All successful variants preserve intended plaintext results; no encrypted accuracy test performed.',flush=True)
