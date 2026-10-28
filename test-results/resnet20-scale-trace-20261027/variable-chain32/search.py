from pathlib import Path
from collections import Counter
import ast
import copy
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import time

ROOT = Path('/home/xuming/poseidon-manual-boot-20261026')
AUDIT = ROOT/'resnet20-optimized-20261027T055506Z/placement-audit'
OUT = AUDIT/'variable-chain32'
OUT.mkdir(exist_ok=True)
BASE = json.loads((AUDIT/'one-relu-full-spec.json').read_text())
Q = BASE['context']['rns_moduli']
SPECIAL = json.loads((ROOT/'src/third_party/ckks-runtime/tests/fixtures/resnet20-manual-q50.json').read_text())['p_moduli']
SOURCE = AUDIT/'one-relu-tagged.mlir'
GENERATOR = ROOT/'src/third_party/ckks-runtime/integrations/dacapo/generate_model_artifacts.py'
COMPILER = ROOT/'hecate-opt'
CONFIG = ROOT/'alignment-tests/compiler.json'
TARGET = sum(math.log2(q) for q in Q[6:31])
STRIDE = 2*BASE['context']['poly_degree']

# Reuse the already checked parser and primality test without running its probes.
module = ast.parse((AUDIT/'chain-waste-probe.py').read_text())
helpers = ast.Module(body=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ('is_prime','extract')],type_ignores=[])
exec(compile(helpers,'chain-waste-probe-helpers','exec'))


def below(limit,used):
    q = (int(limit)-1)//STRIDE*STRIDE+1
    while q in used or not is_prime(q):
        q -= STRIDE
    assert q > STRIDE and q.bit_length() <= 32
    return q


def chain_for(logs):
    assert abs(sum(logs)-TARGET) < 1e-7
    used = set(Q[:6]+Q[31:]+SPECIAL)
    working = []
    for scale in logs:
        q = below(2**scale,used)
        working.append(q)
        used.add(q)
    # Fill prime-rounding loss without increasing the original actual budget.
    loss = TARGET-sum(math.log2(q) for q in working)
    idx = max((i for i,q in enumerate(working) if math.log2(q)+loss<31.9999),key=lambda i:working[i])
    others = sum(math.log2(q) for i,q in enumerate(working) if i != idx)
    used.remove(working[idx])
    working[idx] = below(2**(TARGET-others),used)
    gap = TARGET-sum(math.log2(q) for q in working)
    assert -1e-8 <= gap < .25,(gap,logs)
    result = Q[:6]+working+Q[31:]
    assert max(q.bit_length() for q in result) <= 32 and len(result)==len(set(result))
    return result


def profile(chain,anchor):
    spec = copy.deepcopy(BASE)
    n = len(chain)
    spec['context']['rns_moduli'] = chain
    spec['context']['rns_moduli_log2'] = [q.bit_length() for q in chain]
    spec['context']['max_modulus_log2'] = max(q.bit_length() for q in chain)
    spec['levels']['upper_bound'] = n-1
    for op in spec['operators'].values():
        for key in ('latency_us_by_level','noise_by_level'):
            if isinstance(op.get(key),list):
                original = op[key]
                assert len(set(original)) == 1
                op[key] = [original[0]]*n
    boot = spec['boot_profiles'][0]
    boot['output_level'] = anchor
    boot['latency_us_by_input_level'] = [85000 if i == 5 else 0 for i in range(n)]
    return spec


def alignment(row,chain):
    final = row['stages'][-1]
    q,scale = final['q'],final['scale']
    plain = 40-scale
    needed = 0
    if abs(plain)>1e-8:
        while plain<30:
            needed += 1
            plain += math.log2(chain[5+needed])
    feasible = q>=6+needed
    # Separate comparison with a common s=40 output, retaining maximum Q.
    normal = 40-scale
    drops = 0
    if abs(normal)>1e-8:
        while normal<30 and q-drops>1:
            normal += math.log2(chain[q-drops-1])
            drops += 1
    row.update(boot_alignment_feasible=feasible,boot_alignment_rescales=needed,boot_alignment_plain_scale=plain,normalized_output_q=q-drops,normalized_output_capacity=sum(math.log2(z) for z in chain[:q-drops]),output_normalization_rescales=drops)


results = []
retained = set()
best_capacity = -1
best_rescales = 10**9
t0 = time.monotonic()
cached = {}
if (OUT/'results.json').exists():
    prior = json.loads((OUT/'results.json').read_text())
    assert prior['source_sha256'] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    results = prior['results']
    retained = set(prior['retained'])
    cached = {r['name']:r for r in results}
    valid_prior = [r for r in results if r['status']=='ok']
    best_capacity = max(r['normalized_output_capacity'] for r in valid_prior)
    best_rescales = min(r['counts']['rescale'] for r in valid_prior)
    t0 -= prior['elapsed_seconds']


def score(row):
    if row['status'] != 'ok':
        return (1,1,1,1)
    return (not row['boot_alignment_feasible'],-row['normalized_output_capacity'],row['counts']['rescale'],row['unit_multiplications'])


def save():
    (OUT/'results.json').write_text(json.dumps(dict(max_modulus_bits=32,waterline=40,target_working_actual_log2=TARGET,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-t0,retained=sorted(retained),results=results),indent=2)+'\n')


def evaluate(name,chain,logs=None):
    global best_capacity,best_rescales,retained
    case = OUT/name
    if name in cached and (name not in retained or case.exists()):
        assert cached[name]['moduli'] == chain
        return cached[name]
    if name in cached:
        results.remove(cached[name])
    case.mkdir(exist_ok=True)
    anchor = len(chain)-20
    spec = profile(chain,anchor)
    path = case/'operator-spec.json'
    path.write_text(json.dumps(spec,indent=2)+'\n')
    row = dict(name=name,status='pending',input_q=anchor+1,moduli=chain,working_actual_log2=sum(math.log2(q) for q in chain[6:anchor+1]),full_actual_log2=sum(math.log2(q) for q in chain),working_bits=[q.bit_length() for q in chain[6:anchor+1]],target_logs=logs)
    start = time.monotonic()
    command = ['python3',str(GENERATOR),'mlp','--operator-spec',str(path),'--compiler-profile',str(CONFIG),'--hecate-opt',str(COMPILER),'--traced-mlir',str(SOURCE),'--output-dir',str(case/'artifacts'),'--device-counts','1','--plan-format','json','--boot-placement','greedy','--waterline','40']
    run = subprocess.run(command,capture_output=True,text=True)
    row['seconds'] = time.monotonic()-start
    (case/'compile.log').write_text(run.stdout+run.stderr)
    if run.returncode:
        row.update(status='rejected',error=(run.stdout+run.stderr)[-2500:])
    else:
        plan = json.loads(next((case/'artifacts').glob('*.runtime-plan.json')).read_text())
        boots = sum(op.get('op') == 'boot' for op in plan['execution'])
        if boots:
            row.update(status='internal-boot',boots=boots)
        else:
            row.update(extract(plan,(case/'artifacts/mlp.optimized.mlir').read_text(),chain,anchor))
            row.pop('multiplications')
            row.update(status='ok',boots=0)
            alignment(row,chain)
    results.append(row)
    valid = [r for r in results if r['status']=='ok']
    selected = sorted(valid,key=score)[:6]
    selected += sorted(valid,key=lambda r:(r['counts']['rescale'],r['unit_multiplications']))[:3]
    selected += [results[0]]
    keep = {r['name'] for r in selected}
    for old in (retained|{name})-keep:
        shutil.rmtree(OUT/old)
    retained = keep
    if row['status']=='ok' and (row['normalized_output_capacity']>best_capacity or row['counts']['rescale']<best_rescales):
        best_capacity = max(best_capacity,row['normalized_output_capacity'])
        best_rescales = min(best_rescales,row['counts']['rescale'])
        print(json.dumps(dict(event='improvement',name=name,input_q=row['input_q'],q_consumed=row['total_q_consumed'],rescale=row['counts']['rescale'],unit=row['unit_multiplications'],normalized_q=row['normalized_output_q'],boot_fit=row['boot_alignment_feasible'],stage_consumption=[s['consumed_q'] for s in row['stages']],working_bits=row['working_bits']),ensure_ascii=False),flush=True)
    save()
    if len(results)%16==0:
        print('progress',len(results),'seconds',round(time.monotonic()-t0,1),'best',sorted(valid,key=score)[0]['name'],flush=True)
    return row


def transfer(logs,rng,count,amount):
    logs = logs[:]
    for _ in range(count):
        i,j = rng.sample(range(len(logs)),2)
        delta = min(rng.random()*amount,31.995-logs[i],logs[j]-24.05)
        if delta>0:
            logs[i] += delta
            logs[j] -= delta
    return logs


evaluate('manual',Q)
for count in (24,25,26,27):
    logs = [TARGET/count]*count
    evaluate(f'uniform-{count}',chain_for(logs),logs)

patterns = [
    [32]*3+[30]*22,
    [32]*14+[28]*11,
    [32]*19+[24]*5+[28],
    [31]*19+[28]*5+[27],
    [31]*12+[32]*12,
]
for index,bits in enumerate(patterns):
    # Keep the width allocation but scale its actual total back to the baseline.
    logs = [b-(sum(bits)-TARGET)/len(bits) for b in bits]
    for order in range(6):
        candidate = logs[:]
        random.Random(index*100+order).shuffle(candidate)
        evaluate(f'pattern-{index}-{order}',chain_for(candidate),candidate)

for count,number in ((24,48),(25,96),(26,32),(27,16)):
    for seed in range(number):
        rng = random.Random(count*10000+seed)
        logs = transfer([TARGET/count]*count,rng,50+seed%4*100,.5+seed%5)
        evaluate(f'random-{count}-{seed}',chain_for(logs),logs)

# Local changes around promising chains, still testing the real compiler.
seeds = sorted([r for r in results if r['status']=='ok' and r['target_logs']],key=score)[:4]
for index,seed in enumerate(seeds):
    for step in range(16):
        rng = random.Random(800000+index*100+step)
        logs = transfer(seed['target_logs'],rng,2+step%4,.01 if step<8 else .2)
        evaluate(f'refine-{index}-{step}',chain_for(logs),logs)

for name in sorted(retained):
    case = OUT/name
    plan = next((case/'artifacts').glob('*.runtime-plan.json'))
    verify = subprocess.run([str(ROOT/'validate-plan'),str(plan),str(case/'operator-spec.json')],capture_output=True,text=True)
    (case/'validation.log').write_text(verify.stdout+verify.stderr)
    assert verify.returncode==0,verify.stdout+verify.stderr
    next(r for r in results if r['name']==name)['cpp_validated'] = True
save()
print(json.dumps(dict(event='finished',candidates=len(results),status_counts=dict(Counter(r['status'] for r in results)),elapsed_seconds=time.monotonic()-t0,retained=sorted(retained)),ensure_ascii=False),flush=True)
