#!/usr/bin/env python3
"""Read-only, standard-library audit of archived Cerium compiled function DAGs.

No Cerium/Torch/CUDA imports, code generation, execution, or file writes.
Print deterministic JSON, or --check an existing audit JSON. Refresh counts
come from recursively expanding functions0; [X] output bindings are discarded
limbs, not retained CT streams. Static numerical contracts are source-audited
annotations, not values inferred from the function-name strings.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

ARCHIVE = 'test-results/bert-a800-20261114/cerium/fullblock'
COMPILED = ARCHIVE + '/bert_outputs/compiled'
B33 = 'bootstrap_32K_33lvl_t1'
B35 = 'bootstrap_32K_35lvl_t1_vec_1'
ENTRY = 'main_1gpu_1'
STAGE_NAMES = {
    'attention_qkv_1gpu': 'QKV',
    'matmul_ct128x128_ct128x128_transpose': 'QK',
    'softmax_128x128_vec': 'Softmax',
    'matmul_ct128x128_ct128x128': 'AV',
    'attention_o_1gpu': 'O_and_residual',
    'layernorm_att_1gpu': 'attLN',
    'ffn_up_1gpu': 'FFNup', 'gelu_vec': 'GELU',
    'ffn_down_1gpu': 'down_and_residual',
    'layernorm_ffn_1gpu': 'ffnLN',
}
# Inputs/checkpoints are (active Q prime COUNT, exact-scale alias). They are
# audited from the Python schedule AND the runner/parameter encoder scales.
CONTRACTS = [
    ('QKV', [128,768], 3, [6,6,6], 'two gapped 128x64 heads per Q/K/V CT',
     [(13,'H')], [(9,'Q'),(9,'K'),(9,'Q')]),
    ('QK', [12,128,128], 12, [6], 'two cyclic-diagonal 128x128 heads per CT',
     [(9,'Q'),(9,'K')], [(4,'A/1024'),(16,'A')]),
    ('Softmax', [12,128,128], 6, [6], 'same cyclic-diagonal layout; three vec2 calls',
     [(16,'A')], [(8,'P')]),
    ('AV', [128,768], 12, [6,3], 'gapped head pairs, then pairwise rotate(-64) stacking',
     [(8,'P'),(9,'Q')], [(5,'R')]),
    ('O_and_residual', [128,768], 3, [3], 'three [128 tokens,256 features] CTs; plus H residual',
     [(5,'R'),(13,'H')], [(3,'D'),(2,'B/1024'),(18,'B')]),
    ('attLN', [128,768], 3, [3], 'same hidden layout; one variance/isqrt CT broadcast by token',
     [(18,'B')], [(5,'R')]),
    ('FFNup', [128,3072], 3, [12], 'twelve [128,256] CTs, dense bias included',
     [(5,'R')], [(2,'U'),(21,'U')]),
    ('GELU', [128,3072], 12, [12], 'six two-CT arithmetic batches, no bootstrap subcalls',
     [(21,'U')], [(5,'R')]),
    ('down_and_residual', [128,768], 12, [3], 'dense down; plus three attLN residual CTs',
     [(5,'R')], [(3,'D'),(2,'B/1024'),(18,'B')]),
    ('ffnLN', [128,768], 3, [3], 'raw function output then wrapper extra rescale',
     [(18,'B')], [(3,'p2*T'),(2,'T')]),
    ('final_refresh', [128,768], 3, [3], 'b35 scalar, crop to Q13, multiply by p10*256',
     [(2,'T')], [(13,'T'),(13,'H')]),
]
REFERENCES = [
    (ARCHIVE+'/bert_cerium.py', [97,138], 'QK'),
    (ARCHIVE+'/bert_cerium.py', [222,451], 'Softmax/max/isqrt'),
    (ARCHIVE+'/bert_cerium.py', [534,656], 'AV and QKV'),
    (ARCHIVE+'/bert_cerium.py', [1119,1200], 'attention and residual refresh'),
    (ARCHIVE+'/bert_cerium.py', [1507,1779], 'LN/GELU/up'),
    (ARCHIVE+'/bert_cerium.py', [1942,1986], 'down and residual'),
    (ARCHIVE+'/bert_cerium.py', [2197,2214], 'FFN refresh placement'),
    (ARCHIVE+'/bert_cerium.py', [2519,2565], 'single-block main and final refresh'),
    (ARCHIVE+'/bootstrap.py', [298,544], 'bootstrap cores and scalar/vec2 wrappers'),
    (ARCHIVE+'/bootstrap.py', [688,808], 'ground modulus, Hanki parameters and actual PT scales'),
    (ARCHIVE+'/bert_runner.py', [51,68], 'packing and cyclic mask indices'),
    (ARCHIVE+'/bert_runner.py', [414,652], 'stage decoding scale conventions'),
    (ARCHIVE+'/bert_runner.py', [727,751], 'dense and ephemeral secret weights'),
    (ARCHIVE+'/bert_params_torch.py', [96,250], 'head order and actual encoded parameter scales'),
    (ARCHIVE+'/primes.py', [17,17], 'ordered 64-prime pool'),
    ('../reference/cerium/cerium/compiler/backend/terms.h', [625,726], 'active Q prefix and dynamic P suffix'),
    ('../reference/cerium/cerium/compiler/backend/compiler_keyswitch.cc', [198,250], 'extension_size from decomposition'),
    ('../reference/cerium/cerium/compiler/backend/compiler_rescale.cc', [101,213], 'temporary raise level + 1 then actual rescale'),
    ('third_party/ckks-runtime/runtime/plan.hpp', [30,69], 'standard op/payload boundary'),
    ('src/poseidon/runtime_api/poseidon_gpu_api.cpp', [2161,2229], 'one-CT native Boot contract'),
    ('src/poseidon/gpu/gpu_evaluator.h', [753,761], 'GPU conjugate exists below RuntimePlan'),
    ('bench/dsl/bert/cerium_packing.py', [1,28], 'new compact scheme keeps our different polynomial workload'),
    ('bench/dsl/bert/cerium_lowering.py', [1,7], 'new two-prime lowerer explicitly is not the Cerium compiler'),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_calls(path, root):
    calls, fields, start, name_line = [], {}, 0, 0
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if re.fullmatch(r'function\d+ \{', line):
            fields, start, name_line = {}, line_no, 0
        elif line == '}' and fields.get('type') == 'call':
            callee = fields['function_name'].split('(', 1)[0]
            call = dict(callee=callee, path=str(path.relative_to(root)),
                        line=name_line, block_start_line=start)
            if callee in (B33, B35):
                live = {s.strip() for s in fields['outputs'].split(',')
                        if s.strip() and '[X]' not in s}
                bindings = re.findall(r'<(z[01]?):c[01]\(\d+\)~([^>]+)>',
                                      fields['function_args'])
                lanes = sorted({n for n, value in bindings if value in live})
                expected = [('z',)] if callee == B33 else [('z0',), ('z0','z1')]
                if tuple(lanes) not in expected:
                    raise ValueError(f'Unrecognized bootstrap output bindings: {call}: {lanes}')
                call.update(retained_output_streams=lanes,
                            logical_single_ct_refreshes=len(lanes),
                            count_kind='b33' if callee == B33 else
                            ('b35_vec2' if len(lanes) == 2 else 'b35_scalar'))
            calls.append(call)
        else:
            match = re.match(r'\s*\.(\w+):\s*(.*)', line)
            if match:
                fields[match[1]] = match[2]
                if match[1] == 'function_name':
                    name_line = line_no
    return calls


def totals(counts):
    b33, scalar, vec2 = (counts[k] for k in ('b33','b35_scalar','b35_vec2'))
    return dict(b33=b33, b35_scalar=scalar, b35_vec2=vec2,
                packed_core_executions=b33+scalar+vec2,
                logical_single_ct_refreshes=b33+scalar+2*vec2)


def audit(root):
    primes_tree = ast.parse((root/ARCHIVE/'primes.py').read_text())
    primes = next(ast.literal_eval(n.value) for n in primes_tree.body
                  if isinstance(n, ast.Assign) and
                  any(isinstance(t, ast.Name) and t.id == 'Primes' for t in n.targets))
    compiled = root/COMPILED
    dags, artifacts = {}, {}
    for path in sorted(compiled.glob('*/functions0')):
        name = path.parent.name
        dags[name] = parse_calls(path, root)
        generated = root/ARCHIVE/'bert_outputs'/name/'partition_0/ll.functions'
        artifacts[name] = dict(path=str(path.relative_to(root)), sha256=sha256(path),
            generated_dag_byte_identical=path.read_bytes() == generated.read_bytes())
    invocations, all_counts, per_stage, sites = Counter(), Counter(), defaultdict(Counter), Counter()

    def walk(name, stage, stack=()):
        if name in stack:
            raise ValueError('Recursive function cycle: ' + ' -> '.join(stack+(name,)))
        invocations[name] += 1
        current_stage, final_ln_seen = stage, False
        for call in dags[name]:
            callee = call['callee']
            if name == ENTRY:
                if callee == B35:
                    current_stage = 'final_refresh' if final_ln_seen else 'QK'
                elif callee != B33:
                    current_stage = STAGE_NAMES[callee]
                    final_ln_seen |= callee == 'layernorm_ffn_1gpu'
            if 'count_kind' in call:
                kind = call['count_kind']
                all_counts[kind] += 1
                per_stage[current_stage][kind] += 1
                sites[(name,call['line'])] += 1
            walk(callee, current_stage, stack+(name,))

    walk(ENTRY, 'entry')
    site_rows = []
    for name in sorted(dags):
        for call in dags[name]:
            if 'count_kind' in call:
                site_rows.append(dict(call, caller=name,
                                      expanded_multiplicity=sites[(name,call['line'])]))
    p = primes
    scales = dict(H=(p[12]*p[11]*p[10], 'p12*p11*p10'),
        Q=(p[8]*p[7]*p[6], 'p8*p7*p6'), K=(p[8]*p[5]*p[4], 'p8*p5*p4'),
        A=(p[15]*p[14], 'p15*p14'), P=(p[7]*p[6]*p[5], 'p7*p6*p5'),
        R=(p[4]*p[3]*p[2], 'p4*p3*p2'), B=(p[17]*p[16], 'p17*p16'),
        U=(p[20]*p[19]/1000, 'p20*p19/1000'),
        D=(p[2]*p[17]*p[16]/1024, 'p2*p17*p16/1024'),
        T=(p[12]*p[11]/256, 'p12*p11/256'))
    stages = []
    for name, shape, input_ct, output_ct, layout, inputs, checkpoints in CONTRACTS:
        stages.append(dict(name=name, semantic_output_shape=shape,
            primary_input_ct_count=input_ct, output_ct_counts=output_ct, layout=layout,
            inputs=[dict(active_q_count=q, scale=s) for q,s in inputs],
            checkpoints=[dict(active_q_count=q, scale=s) for q,s in checkpoints],
            refresh=totals(per_stage[name])))
    sources = [dict(path=path, lines=lines, description=desc)
               for path,lines,desc in REFERENCES]
    for ref in sources:
        if not (root/ref['path']).is_file():
            raise FileNotFoundError(ref['path'])
    return dict(schema_version=1, scope='local static audit; no GPU/runtime imports',
        path_convention='Paths are relative to Poseidon repository root; ../reference is its sibling source tree.',
        archive=ARCHIVE, entry=ENTRY,
        refresh_unit='one independent input CT stream refreshed to one retained output CT stream; vec2 counts two',
        counting_method='Recursively expand compiled functions0 calls; classify z/z0/z1 retained output bindings excluding [X] discarded limbs.',
        totals=totals(all_counts), expanded_function_invocations=dict(sorted(invocations.items())),
        bootstrap_call_sites=site_rows, function_dag_artifacts=artifacts,
        main_so=dict(path=COMPILED+'/'+ENTRY+'/function_map_0.so',
                     sha256=sha256(compiled/ENTRY/'function_map_0.so')),
        scales={name:dict(formula=formula, value=value, log2=math.log2(value))
                for name,(value,formula) in scales.items()}, stages=stages,
        packing=dict(hidden_cts=3, hidden_ct_shape=[128,256], score_cts=6,
            hidden_index='H_c[256*t+f]=X[t,256*c+f]',
            head_order=[0,2,1,3,4,6,5,7,8,10,9,11],
            qkv_index='CT_n[256*t+128*k+f]=head[HEAD_ORDER[2*n+k]][t,f], f<64; other half-slots zero',
            score_index='Score_n[256*t+128*k+d]=dot(Q_h[t,:],K_h[(t+d)%128,:])/40',
            context_stacking='Context[2*c] + rotate(Context[2*c+1], -64)'),
        he_parameters=dict(poly_degree=65536, slots=32768, nominal_scale_bits=28,
            level_definition='L means active Q prime count; Q_L=product(p[0:L]), not Poseidon level index',
            ordered_prime_pool=primes, pool_is_not_active_q=True,
            q_first_prime=p[0], bootstrap_ground_indices=[0,1],
            bootstrap_ground_modulus=p[0]*p[1],
            main_secret_hamming_weight=32768, ephemeral_secret_hamming_weight=32,
            hanki=dict(degree=30,double_angle=3,k=16,epsilon='1/256'),
            b33=dict(raise_q_count=52, temporary_keyswitch_q_count=53, core_output_q_count=21),
            b35=dict(raise_q_count=49, temporary_keyswitch_q_count=50, core_output_q_count=16),
            wrapper_output_q_count='availableLevels + 2',
            special_p=dict(policy='dynamic suffix p[64-e:64], not a fixed P12 or P13; same pool may serve Q at another level',
                examples=[dict(operation=op,active_q_count=q,extension_count=e,
                               pool_indices=list(range(64-e,64)),primes=p[64-e:])
                          for op,q,e in [('b33 rotation',52,11),('b35 rotation',49,13),
                                         ('L13 linear rotation',13,13),('L30 relin',30,30),
                                         ('toEphemeral',2,1),('b33 raise key temporary domain',53,2),
                                         ('b35 raise key temporary domain',50,2)]])),
        known_debug_annotation_issue=dict(path=ARCHIVE+'/bert_runner.py',line=601,
            label='attention_residual_output_bs', recorded='p17*p16',
            actual_pre_refresh='p17*p16/1024',
            derivation='O: R*(B/1024)/(p4*p3)=p2*B/1024; residual rescale /p2; post-Boot scalar encoded scale 1024 restores B'),
        current_poseidon_candidate=dict(hidden_cts=3,score_cts=6,
            label='native56/d30DA3K16 candidate; partial alignment only',
            aligned_subset=['compact packing dimensions','two-prime/high-scale scheduling choice','candidate EvalMod degree/DA/K'],
            not_claimed=['Cerium sign/max Softmax and six-step exp/isqrt','Cerium piecewise GELU','identical Hanki coefficients/epsilon/DFT/P/key policy',
                         'identical true scales/ground modulus/output Q','complex vec2 pairing and conjugation lowering','measured GPU correctness or performance']),
        runtime_boundary=dict(real_linear_and_cc_matmul='existing arithmetic ops suffice; fused performance not implied',
            periodic_real_plaintext='explicitly tile to full slots for semantic replay; compressed encoded storage/remapping not retained',
            conjugation='GPU backend exists, standard RuntimePlan has no Conjugate exposure',
            complex_plaintext='CPU encoder supports it below API; current Plain/EncodePayload/GPU Prepare path is real-only',
            paired_refresh='pack/complex Boot/split needs complex plaintext and conjugation exposure; no standard pairing lowering today'),
        sources=sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--check', type=Path, help='compare against a previously saved audit JSON, without writing')
    args = parser.parse_args()
    root = args.repo.resolve()
    result = audit(root)
    if args.check:
        path = args.check if args.check.is_absolute() else root/args.check
        if json.loads(path.read_text()) != result:
            raise SystemExit('Audit differs: ' + str(path))
        print('OK: ' + json.dumps(result['totals'], sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    main()
