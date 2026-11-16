#!/usr/bin/env python3
"""Build BERT DSL fixtures and explicit scale/Boot RuntimePlans (CPU only)."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from backend import Graph, ManualLowering, digest, json_bytes


def dense_reference(hidden, weights, config, scope):
    """Independent dense polynomial reference for the prefix model."""
    def ln(x, w, prefix):
        centered = x - x.mean(axis=-1, keepdims=True)
        u = (np.mean(centered*centered, axis=-1, keepdims=True)+config.ln_epsilon)/config.ln_variance_center - 1
        inv = (1 - .5*u + .375*u*u - .3125*u*u*u)/np.sqrt(config.ln_variance_center)
        return centered*inv*w[prefix+'_gamma'] + w[prefix+'_beta']
    x = hidden.astype(np.float64)
    if scope == 'ffn-up': return x @ weights[0]['ffn_up_weight'].T + weights[0]['ffn_up_bias']
    for w in weights:
        q, k, v = [(x @ w[name+'_weight'].T+w[name+'_bias']).reshape(config.tokens,config.heads,config.head_dim).transpose(1,0,2)
                   for name in ('q','k','v')]
        s = (q @ k.transpose(0,2,1))/np.sqrt(config.head_dim)
        s -= s.mean(axis=-1, keepdims=True)
        e = 1+s+.5*s*s+s*s*s/6+s*s*s*s/24
        d = e.sum(axis=-1, keepdims=True)
        inv = np.full_like(d, 1/config.tokens)
        for _ in range(config.reciprocal_iterations): inv *= 2-d*inv
        attention = ((e*inv) @ v).transpose(1,0,2).reshape(config.tokens,config.hidden)
        x = ln(attention @ w['o_weight'].T+w['o_bias']+x, w, 'attention_ln')
        up = x @ w['ffn_up_weight'].T+w['ffn_up_bias']
        gelu = .5*up+(up*up-up**4/6)/np.sqrt(2*np.pi)
        x = ln(gelu @ w['ffn_down_weight'].T+w['ffn_down_bias']+x,w,'ffn_ln')
    return x


def emit_earth(graph, outputs, path):
    """Export the same frontend arithmetic as standard Hecate Earth MLIR.

    The automatic --dacapo pipeline intentionally replans Earth boot hints.
    Use the explicit RuntimePlan (or physical CKKS checkpoint) to preserve them.
    """
    ct='tensor<1x!earth.ci<0 * 0>>'; pt='tensor<1x!earth.pl<0 * 0>>'
    nodes=graph.reachable(outputs); names={n.index:f'%n{n.index}' for n in nodes}
    types={n.index: pt if n.op=='plain' else ct for n in nodes}
    args=', '.join(f'{names[n.index]}: {ct}' for n in graph.inputs)
    result=', '.join(ct for _ in outputs)
    with Path(path).open('w') as f:
        f.write(f'module {{ func.func @bert({args}) -> ({result}) {{\n')
        for n in nodes:
            if n.op=='input': continue
            operands=', '.join(names[a.index] for a in n.args)
            in_types=', '.join(types[a.index] for a in n.args)
            attr=''; opcode=n.op
            if n.op=='plain':
                opcode='constant'; a=graph.constants[n.data]
                attr=f' {{value = dense<"0x{a.tobytes().hex().upper()}"> : tensor<{len(a)}xf64>, rms_var = {float(np.mean(a*a)):.17e} : f64}}'
            elif n.op=='rotate': attr=f' {{offset = array<i64: {n.data}>}}'
            elif n.op=='boot': opcode='bootstrap'
            elif n.op=='neg': opcode='negate'
            elif n.op=='sub':
                a,b=n.args
                neg=f'%neg{n.index}'
                if b.op=='plain':
                    values=-graph.constants[b.data]
                    f.write(f'{neg} = "earth.constant"() {{value = dense<"0x{values.tobytes().hex().upper()}"> : tensor<{len(values)}xf64>, rms_var = {float(np.mean(values*values)):.17e} : f64}} : () -> {pt}\n')
                else:
                    f.write(f'{neg} = "earth.negate"({names[b.index]}) : ({types[b.index]}) -> {types[b.index]}\n')
                operands=f'{names[a.index]}, {neg}'; opcode='add'
            f.write(f'{names[n.index]} = "earth.{opcode}"({operands}){attr} : ({in_types}) -> {types[n.index]}\n')
        f.write('return '+', '.join(names[o.index] for o in outputs)+' : '+result+'\n}}\n')


def emit_ckks(plan, graph, q, path):
    """Physical checkpoint accepted by --ckks-to-runtime-plan, no replanning."""
    desc={v['id']:v for v in plan['values']}
    def ty(i):
        v=desc[i]; return f'tensor<1x!ckks.poly<{v["components"]} * {int(np.ceil(v["scale_log2"]))} * {v["level"]}>>'
    names={i:'%v'+i for i in desc}
    contents={}
    final=plan['final_outputs']; ext=plan['external_inputs']
    with Path(path).open('w') as f:
        f.write('module { func.func @bert('+', '.join(names[i]+': '+ty(i) for i in ext)+') -> ('+', '.join(ty(i) for i in final)+') attributes {')
        f.write('earth.arg_scales = array<f64: '+', '.join(f'{desc[i]["scale_log2"]:.17e}' for i in ext)+'>, ')
        f.write('ckks.rns_moduli = array<i64: '+', '.join(map(str,q))+'>} {\n')
        mapping=dict(add_cc='addcc',add_cp='addcp',sub_cc='subcc',sub_cp='subcp',mul_cc='mulcc',mul_cp='mulcp',
                     negate='negatec',rotate='rotatec',rescale='rescalec',mod_switch='modswitchc',relinearize='relinearize',boot='bootstrapc')
        for row in plan['initialization']+plan['execution']:
            if row['kind']=='release': continue
            if row['kind']=='transfer':
                for a,b in zip(row['inputs'],row['outputs']):
                    names[b]=names[a]
                    if a in contents: contents[b]=contents[a]
                continue
            out=row['output']; scale=desc[out]['scale_log2']
            attr=[f'runtime.scale_log2 = {scale:.17e} : f64']
            inputs=row.get('inputs',[])
            if row['kind']=='encode':
                opcode='encode'; contents[out]=row['payload']['content']
                a=graph.constants[contents[out]]
                attr.append(f'payload = dense<"0x{a.tobytes().hex().upper()}"> : tensor<{len(a)}xf64>')
            else:
                opcode=mapping[row['op']]
                if row['op'] in ('sub_cc','sub_cp'):
                    left,right=inputs
                    neg='%neg'+out
                    if row['op']=='sub_cp':
                        values=-graph.constants[contents[right]]
                        f.write(f'{neg} = "ckks.encode"() {{runtime.scale_log2 = {desc[right]["scale_log2"]:.17e} : f64, payload = dense<"0x{values.tobytes().hex().upper()}"> : tensor<{len(values)}xf64>}} : () -> {ty(right)}\n')
                    else:
                        f.write(f'{neg} = "ckks.negatec"({names[right]}) : ({ty(right)}) -> {ty(right)}\n')
                    f.write(f'{names[out]} = "ckks.{"addcp" if row["op"]=="sub_cp" else "addcc"}"({names[left]}, {neg}) {{runtime.scale_log2 = {scale:.17e} : f64}} : ({ty(left)}, {ty(right)}) -> {ty(out)}\n')
                    continue
                if row['op']=='rotate': attr.append(f'offset = array<i64: {row["attrs"]["steps"]}>')
                if row['op']=='mod_switch': attr.append(f'downFactor = {desc[inputs[0]]["level"]-desc[out]["level"]} : i64')
            f.write(f'{names[out]} = "ckks.{opcode}"('+', '.join(names[i] for i in inputs)+') {'+', '.join(attr)+'} : ('+', '.join(ty(i) for i in inputs)+') -> '+ty(out)+'\n')
        f.write('return '+', '.join(names[i] for i in final)+' : '+', '.join(ty(i) for i in final)+'\n}}\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--layout',choices=('prefix','packed'),default='prefix')
    p.add_argument('--tokens',type=int,default=2); p.add_argument('--layers',type=int,default=1)
    p.add_argument('--hidden',type=int,default=768); p.add_argument('--intermediate',type=int,default=3072)
    p.add_argument('--heads',type=int,default=12); p.add_argument('--slots',type=int,default=32768)
    p.add_argument('--row-stride',type=int,default=4096)
    p.add_argument('--scope',choices=('encoder','ffn-up'),default='encoder')
    p.add_argument('--strategy',choices=('stage','budget','selective'),default='selective')
    p.add_argument('--no-fuse',action='store_true'); p.add_argument('--resident',action='store_true')
    p.add_argument('--q-count',type=int,default=32); p.add_argument('--p-count',type=int,default=16)
    p.add_argument('--boot-output-level',type=int,default=16); p.add_argument('--boot-input-min',type=int,default=2)
    p.add_argument('--boot-scale',type=int,default=30)
    p.add_argument('--input-level',type=int,help='manual lower initial Q level, useful for isolated linear layers')
    p.add_argument('--profile-report',type=Path,help='bind the actual native --probe-profile report')
    p.add_argument('--seed',type=int,default=20261114)
    p.add_argument('--emit-earth',action='store_true'); p.add_argument('--emit-ckks',action='store_true')
    a=p.parse_args(); start=time.monotonic()
    if a.output_dir.exists(): p.error('--output-dir must not exist')
    if a.hidden % a.heads: p.error('hidden must be divisible by heads')
    probe=None
    if a.profile_report:
        probe=json.loads(a.profile_report.read_bytes())
        if not probe.get('completed') or not probe.get('profile_probe_only'):
            p.error('--profile-report must be a completed real native profile probe')
        if probe['context']['poly_degree'] != 2*a.slots:
            p.error('probe degree does not match requested slots')
        actual=probe['native_boot_profile']
        if actual['input_components'] != 2 or actual['output_components'] != 2:
            p.error('only two-component native Boot profiles are supported')
    from model import BertConfig, create_weights, build_model, linear
    fields=dict(tokens=a.tokens,layers=a.layers,hidden=a.hidden,intermediate=a.intermediate,
                heads=a.heads,head_dim=a.hidden//a.heads,slots=a.slots)
    if a.layout=='prefix': config=BertConfig(**fields)
    else:
        from packed_model import PackedConfig
        fields.pop('head_dim')
        config=PackedConfig(**fields,row_stride=a.row_stride)
    weights=create_weights(config,a.seed)
    rng=np.random.default_rng(a.seed+1)
    hidden=rng.standard_normal((a.tokens,a.hidden)); hidden/=np.sqrt(np.mean(hidden**2,axis=-1,keepdims=True))
    graph=Graph(a.slots)
    if a.layout=='prefix':
        packed=np.zeros((a.tokens,a.slots)); packed[:,:a.hidden]=hidden
    else:
        rows=a.slots//a.row_stride
        packed=np.zeros((a.tokens//rows,a.slots))
        for token in range(a.tokens):
            packed[token//rows,(token%rows)*a.row_stride:(token%rows)*a.row_stride+a.hidden]=hidden[token]
    inputs=[graph.input('hidden_'+str(i)) for i in range(len(packed))]
    if a.scope=='ffn-up':
        if a.layout=='prefix': outputs=[linear(graph,x,weights[0]['ffn_up_weight'],weights[0]['ffn_up_bias'],slots=a.slots) for x in inputs]
        else:
            from packed_model import linear as packed_linear
            outputs=[packed_linear(graph,x,weights[0]['ffn_up_weight'],weights[0]['ffn_up_bias'],config=config) for x in inputs]
    elif a.layout=='prefix': outputs=build_model(graph,inputs,weights,config,boot_policy='stage')
    else:
        from packed_model import build_model as packed_build
        outputs=packed_build(graph,inputs,weights,config,boot_policy='stage')
    context=dict(poly_degree=2*a.slots,q_moduli_bits=[31]+[30]*(a.q_count-1),
                 p_moduli_bits=[31]*a.p_count,default_scale_log2=30)
    profile=dict(profile_id='bert-native',input_level_min=a.boot_input_min,input_level_max=a.q_count-1,
                 output_level=a.boot_output_level,output_scale_log2=a.boot_scale)
    native_boot=dict(eval_mod_log_scale=a.boot_scale,eval_mod_sine_degree=58,
                     eval_mod_generation_degree=59,eval_mod_double_angle=2)
    if probe:
        context=probe['context']; actual=probe['native_boot_profile']
        profile={k:actual[k] for k in ('profile_id','input_level_min','input_level_max','output_level','output_scale_log2')}
        native_boot=actual['config']
    schedule=ManualLowering(graph,context,profile,a.strategy,not a.no_fuse,initial_level=a.input_level)
    if probe and schedule.q != probe['actual_q_moduli']:
        raise ValueError('probe actual primes disagree with deterministic CoeffModulus::Create')
    stats=schedule.export(outputs,a.output_dir,resident=a.resident)
    stats['native_profile_probe_sha256']=digest(a.profile_report.read_bytes()) if probe else None
    stats['native_profile_verified']=probe is not None
    (a.output_dir/'schedule.json').write_bytes(json_bytes(stats))
    if probe: (a.output_dir/'native-profile-probe.json').write_bytes(a.profile_report.read_bytes())
    reference=dense_reference(hidden,weights,config,a.scope)
    if a.layout=='prefix': expected=reference
    else:
        expected=np.zeros_like(packed)
        for token in range(a.tokens):
            expected[token//rows,(token%rows)*a.row_stride:(token%rows)*a.row_stride+reference.shape[1]]=reference[token]
    fixture=dict(format_version=1,model=dict(name='synthetic-bert-base-encoder',scope=a.scope,layout=a.layout,
                config=asdict(config),seed=a.seed),context=context,
                native_boot=native_boot,
                tolerances=dict(atol=.001,rtol=.0001),
                inputs=[dict(name='hidden_'+str(i),values=row.tolist()) for i,row in enumerate(packed)],
                expected_outputs=[dict(name='encoder_'+str(i),values=row.tolist()) for i,row in enumerate(expected)],
                reference_semantics='dense local polynomial oracle; not pretrained BERT accuracy')
    (a.output_dir/'fixture.json').write_bytes(json_bytes(fixture))
    np.save(a.output_dir/'input_hidden.npy',hidden); np.save(a.output_dir/'reference_hidden.npy',reference)
    if a.emit_earth: emit_earth(graph,outputs,a.output_dir/'frontend.earth.mlir')
    if a.emit_ckks:
        plan=json.loads((a.output_dir/'model.runtime-plan.json').read_bytes())
        emit_ckks(plan,graph,schedule.q,a.output_dir/'manual.ckks.mlir')
    binding={str(path.name):digest(path.read_bytes()) for path in Path(__file__).parent.glob('*.py')}
    try:
        binding['qwen_origin_benchmark_revision']=subprocess.check_output(
            ['git','rev-parse','origin/benchmark'],text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        binding['qwen_origin_benchmark_revision']=None
    if probe: binding['native_profile_probe_sha256']=stats['native_profile_probe_sha256']
    binding['elapsed_seconds']=time.monotonic()-start
    (a.output_dir/'source-binding.json').write_bytes(json_bytes(binding))
    print(json.dumps(dict(output=str(a.output_dir),elapsed_seconds=time.monotonic()-start,**stats),indent=2))


if __name__=='__main__': main()
