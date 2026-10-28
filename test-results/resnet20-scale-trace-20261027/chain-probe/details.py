from collections import defaultdict
from pathlib import Path
import json
import math
import re

audit = Path('/home/xuming/poseidon-manual-boot-20261026/resnet20-optimized-20261027T055506Z/placement-audit')
root = audit/'chain-waste-probe'
original = {o['id']:o for o in json.loads((audit/'scale-trace/scale-trace.json').read_text())['compiler']['operations']}
labels = {}
cases = {}
for case in ('manual','shuffle-only-30-0','working-matched-budget-40'):
    path = root/case/'artifacts'
    plan = json.loads(next(path.glob('*.runtime-plan.json')).read_text())
    values = {v['id']:v for v in plan['values']}
    meta = {}
    for line in (path/'mlp.optimized.mlir').read_text().splitlines():
        identifier = re.search(r'dist.logical_id = (\d+)',line)
        tags = re.findall(r'"raw-(\d+)"',line)
        if not identifier:
            continue
        oid = identifier[1]
        scalar = re.search(r'payload = dense<([-+0-9.eE]+)>',line)
        meta[oid] = dict(raw=tags[0] if tags else None,scalar=float(scalar[1]) if scalar else None)
        if case == 'manual' and tags and oid in original:
            labels.setdefault(tags[0],original[oid]['stage']+' '+original[oid]['label'])
    producers = {}
    plain = {}
    core = {}
    rescales = defaultdict(list)
    for op in plan['initialization']+plan['execution']+plan['finalization']:
        kind = op.get('op',op['kind'])
        out = op.get('outputs',[op['output']] if 'output' in op else [])
        if not out:
            continue
        oid = out[0]
        producers[oid] = op
        args = op.get('inputs',[])
        if kind == 'encode':
            plain[oid] = meta[oid]['scalar']
        elif kind == 'transfer':
            plain[oid] = plain.get(args[0])
        elif kind == 'mul_cc' or (kind == 'mul_cp' and plain[args[1]] != 1):
            key = meta[oid]['raw']+':'+kind
            assert key not in core
            core[key] = dict(id=oid,kind=kind,coefficient=plain.get(args[1]) if kind=='mul_cp' else None,input_scales=[values[a]['scale_log2'] for a in args],product_scale=values[oid]['scale_log2'])
        elif kind == 'rescale':
            before,after = values[args[0]],values[oid]
            rescales[meta[oid]['raw']].append(dict(id=oid,from_q=before['level']+1,to_q=after['level']+1,from_scale=before['scale_log2'],to_scale=after['scale_log2']))
    assert len(core) == 57
    core_ids = {op['id']:key for key,op in core.items()}
    cache = {}
    def ancestors(oid):
        if oid in cache:
            return cache[oid]
        if oid in core_ids:
            result = {core_ids[oid]}
        else:
            result = set().union(*(ancestors(p) for p in producers[oid].get('inputs',[]))) if oid in producers else set()
        cache[oid] = result
        return result
    for key,op in core.items():
        op['nearest_core_parents'] = sorted(set().union(*(ancestors(a) for a in producers[op['id']]['inputs'])))
    cases[case] = dict(core=core,rescales=dict(rescales))
baseline = cases['manual']
comparisons = {}
for name,case in cases.items():
    assert case['core'].keys() == baseline['core'].keys()
    for key,op in case['core'].items():
        ref = baseline['core'][key]
        assert op['coefficient'] == ref['coefficient']
        assert op['nearest_core_parents'] == ref['nearest_core_parents']
    changes = []
    if name == 'manual':
        continue
    for tag in sorted(set(baseline['rescales'])|set(case['rescales']),key=int):
        left,right = baseline['rescales'].get(tag,[]),case['rescales'].get(tag,[])
        if len(left) != len(right):
            changes.append(dict(raw=tag,label=labels.get(tag),baseline=left,variant=right,extra_rescales_in_baseline=len(left)-len(right)))
    comparisons[name] = dict(matched_core_nodes=57,coefficient_values_identical=True,core_dependencies_identical=True,rescale_count_changes=changes)
(root/'differences.json').write_text(json.dumps(dict(comparisons=comparisons,cases=cases,labels=labels),ensure_ascii=False,indent=2)+'\n')
print('Three plans: all 57 arithmetic multiplication nodes, coefficients, and dependencies match.')
for name,case in comparisons.items():
    print(name,'net rescale savings:',sum(x['extra_rescales_in_baseline'] for x in case['rescale_count_changes']))
