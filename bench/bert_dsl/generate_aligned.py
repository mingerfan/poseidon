#!/usr/bin/env python3
"""Generate compact/cyclic BERT and explicit multi-prime schedules (CPU only)."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np

from backend import digest, json_bytes
from cerium_lowering import CompactGraph, MultiPrimeLowering
from cerium_packing import Config, build_model, create_weights, pack_inputs
from generate import dense_reference

# Cerium's ordered pool, NOT a declaration that all 64 primes are active Q.
CERIUM_PRIMES = [
    204865537,205651969,206307329,207880193,209059841,210370561,211025921,211812353,
    214171649,215482369,215744513,216137729,216924161,217317377,218628097,219676673,
    220594177,221249537,222035969,222167041,222953473,223215617,224002049,224133121,
    225574913,228065281,228458497,228720641,230424577,230686721,230817793,231473153,
    232390657,232652801,234356737,235798529,236584961,236716033,239337473,239861761,
    240648193,241827841,244842497,244973569,245235713,245760001,246415361,249561089,
    253100033,253493249,254279681,256376833,256770049,257949697,258605057,260571137,
    260702209,261488641,261881857,263323649,263454721,264634369,265420801,268042241,
]


def candidate(name, bootstrap_ratio=256):
    if name == 'q52-p11':
        q_count, p_count, output, groups = 52, 11, 20, [5, 5, 5]
    elif name == 'q49-p13':
        q_count, p_count, output, groups = 49, 13, 15, [4, 4, 4, 3]
    else:
        raise ValueError(name)
    context = dict(poly_degree=65536, q_moduli_bits=[28] * q_count,
        p_moduli_bits=[28] * p_count, q_moduli=CERIUM_PRIMES[:q_count],
        p_moduli=CERIUM_PRIMES[-p_count:], default_scale_log2=28,
        secret_hamming_weight=32768, q0_level=1)
    if bootstrap_ratio < 1 or bootstrap_ratio & (bootstrap_ratio - 1):
        raise ValueError('bootstrap ratio must be a positive power of two')
    profile = dict(profile_id='bert-aligned-' + name + '-r' + str(bootstrap_ratio), input_level_min=2,
        input_level_max=q_count - 1, output_level=output, output_scale_log2=56)
    native = dict(profile_id=profile['profile_id'], eval_mod_log_scale=56,
        eval_mod_sine_degree=30, eval_mod_generation_degree=31,
        eval_mod_double_angle=3, eval_mod_k=16, eval_mod_log_split=2,
        logical_rescale_count=2, c2s_layer_groups=groups, bootstrap_ratio=bootstrap_ratio,
        sparse_secret_hamming_weight=32, eval_mod_full_coefficients=True,
        compress_linear_plaintexts=True, project_real=True, output_level=output)
    return context, profile, native


# Explicit defaults of this bench runner and its matching profile builder.
# They are used ONLY with a SHA-bound original fixture for legacy reports.
BOOT_DEFAULTS = dict(bootstrap_ratio=32, eval_mod_log_scale=28,
    eval_mod_sine_degree=58, eval_mod_generation_degree=59, eval_mod_double_angle=2,
    eval_mod_k=25, eval_mod_arcsine_degree=0, eval_mod_truncate_degree=None,
    eval_mod_fixed_degree_refit=False, eval_mod_log_split=2,
    eval_mod_virtual_degree_bound=True, logical_rescale_count=1,
    c2s_log_bsgs_ratio=1, c2s_layer_groups=[], c2s_direct_layer_threshold=0,
    s2c_log_bsgs_ratio=1, project_real=False, sparse_secret_hamming_weight=0,
    eval_mod_full_coefficients=False, compress_linear_plaintexts=False, output_level=None)


def bind_native_config(probe, context, original_fixture=None):
    """Never fill incomplete probe choices from a different candidate."""
    actual = probe['native_boot_profile']
    reported = dict(actual['config'])
    if set(BOOT_DEFAULTS) <= reported.keys():
        native = {key: reported[key] for key in BOOT_DEFAULTS}
    else:
        if original_fixture is None:
            raise ValueError('legacy probe omits profile choices; provide its SHA-bound --profile-fixture')
        raw = Path(original_fixture).read_bytes()
        if digest(raw) != probe.get('fixture_sha256'):
            raise ValueError('original probe fixture SHA256 mismatch')
        fixture = json.loads(raw)
        if fixture.get('context') != context:
            raise ValueError('original probe fixture context mismatch')
        native = dict(BOOT_DEFAULTS)
        native['eval_mod_log_scale'] = context['default_scale_log2']
        native.update(fixture.get('native_boot', {}))
        native.update(fixture.get('native_boot_profiles', {}).get(actual['profile_id'], {}))
        for key, value in reported.items():
            if key != 'profile_id' and native.get(key) != value:
                raise ValueError('probe and original fixture disagree on ' + key)
    native['profile_id'] = actual['profile_id']
    return native


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--candidate', choices=('q52-p11', 'q49-p13'), default='q52-p11')
    parser.add_argument('--bootstrap-ratio', type=int, default=256,
                        help='power-of-two message ratio; 256 corresponds to epsilon=2^-8')
    parser.add_argument('--scope', choices=('encoder', 'ffn-up'), default='encoder')
    parser.add_argument('--attention-layout', choices=('cyclic', 'offset'), default='cyclic')
    parser.add_argument('--strategy', choices=('stage', 'budget', 'selective'), default='selective')
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--input-level', type=int, default=12, help='Poseidon level index, Q count minus one')
    parser.add_argument('--seed', type=int, default=20261114)
    parser.add_argument('--profile-report', type=Path)
    parser.add_argument('--periodic-encoding', action='store_true',
                        help='opt in to CPU-validated small-ring preparation; strict encoding is default')
    parser.add_argument('--algebra-rewrite', action='store_true',
                        help='opt in to equivalent full-slot scalar/Newton seed rewrites; re-lower physical schedule')
    parser.add_argument('--shared-rotation-rescale', action='store_true',
                        help='experimentally rescale a reused pending source once before its rotations')
    parser.add_argument('--prepared-qk', action='store_true',
                        help='experimental native prepared cyclic QK; requires matching backend build')
    parser.add_argument('--qk-pack-inputs', action='store_true',
                        help='experimental pre-CC packed-input QK on proven ideal zero head gaps; requires --prepared-qk')
    parser.add_argument('--prepared-av', action='store_true',
                        help='experimental wide-mask AV; fixed q52 selective L10/L8 -> L4 backend')
    parser.add_argument('--av-pack-inputs', action='store_true',
                        help='experimental defined partial AV with masked reciprocal; requires --prepared-av --algebra-rewrite')
    parser.add_argument('--data-from-plan', type=Path,
                        help='bind original client input and exact dense weights; verify all parameter hashes')
    parser.add_argument('--data-prefix-layers', action='store_true',
                        help='explicitly bind a shorter --layers prefix of --data-from-plan for chain diagnostics')
    parser.add_argument('--polynomial-reassociation', action='store_true',
                        help='experimental same-coefficient exp4/GELU4/rsqrt3 reassociation; degree unchanged')
    parser.add_argument('--profile-fixture', type=Path,
                        help='original SHA-bound fixture, required for incomplete legacy probe reports')
    parser.add_argument('--profile-fixture-only', action='store_true')
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error('--output-dir must not exist')
    if args.av_pack_inputs and not args.prepared_av:
        parser.error('--av-pack-inputs requires --prepared-av')
    if args.av_pack_inputs and not args.algebra_rewrite:
        parser.error('--av-pack-inputs requires --algebra-rewrite')
    if args.av_pack_inputs and (args.scope != 'encoder' or args.attention_layout != 'cyclic' or
                               args.candidate != 'q52-p11' or args.input_level != 12 or
                               args.strategy != 'selective'):
        parser.error('--av-pack-inputs requires cyclic selective q52-p11 encoder at input L12')
    if args.qk_pack_inputs and not args.prepared_qk:
        parser.error('--qk-pack-inputs requires --prepared-qk')
    if args.qk_pack_inputs and (args.scope != 'encoder' or args.attention_layout != 'cyclic' or
                               args.candidate != 'q52-p11' or args.input_level != 12 or
                               args.strategy != 'selective'):
        parser.error('--qk-pack-inputs requires cyclic selective q52-p11 encoder at input L12')
    if args.prepared_qk and (args.scope != 'encoder' or args.attention_layout != 'cyclic'):
        parser.error('--prepared-qk requires cyclic encoder scope')
    if args.prepared_av and (args.scope != 'encoder' or args.attention_layout != 'cyclic' or
                            args.candidate != 'q52-p11' or args.input_level != 12 or
                            args.strategy != 'selective'):
        parser.error('--prepared-av requires cyclic selective q52-p11 encoder at input L12')
    if args.data_prefix_layers and not args.data_from_plan:
        parser.error('--data-prefix-layers requires --data-from-plan')
    if args.data_from_plan and (args.scope != 'encoder' or args.profile_fixture_only):
        parser.error('--data-from-plan requires an encoder inference plan')
    context, profile, native = candidate(args.candidate, args.bootstrap_ratio)
    probe = None
    if args.profile_report:
        probe = json.loads(args.profile_report.read_bytes())
        if not probe.get('completed') or not probe.get('profile_probe_only'):
            parser.error('profile report must be a completed native profile probe')
        if probe['context'] != context:
            parser.error('profile context disagrees with candidate')
        actual = probe['native_boot_profile']
        profile = {key: actual[key] for key in (
            'profile_id', 'input_level_min', 'input_level_max', 'output_level', 'output_scale_log2')}
        if probe['actual_q_moduli'] != context['q_moduli'] or probe['actual_p_moduli'] != context['p_moduli']:
            parser.error('actual native Q/P primes disagree')
        try:
            native = bind_native_config(probe, context, args.profile_fixture)
        except ValueError as exc:
            parser.error(str(exc))
        if native['bootstrap_ratio'] != args.bootstrap_ratio:
            parser.error('probe bootstrap ratio differs; select the matching --bootstrap-ratio explicitly')
    if args.profile_fixture_only:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / 'fixture.json').write_bytes(json_bytes(dict(format_version=1,
            context=context, native_boot=native, model=dict(profile_probe_only=True))))
        return
    start = time.monotonic()
    config = Config(layers=args.layers, attention_layout=args.attention_layout)
    weights = create_weights(config, args.seed)
    data_binding = None
    if args.data_from_plan:
        from weight_binding import bind_frozen_weights, load_frozen_input
        try:
            weights, data_binding = bind_frozen_weights(weights, config, args.seed, args.data_from_plan,
                allow_prefix_layers=args.data_prefix_layers)
            hidden, input_hash = load_frozen_input(config, args.data_from_plan,
                expected_fixture_sha256=data_binding['fixture_sha256'],
                allow_prefix_layers=args.data_prefix_layers)
        except (ValueError, OSError) as exc:
            parser.error('fixed data binding: ' + str(exc))
        data_binding.update(original_client_input_loaded=True, input_sha256=input_hash)
    else:
        hidden = np.random.default_rng(args.seed + 1).standard_normal((config.tokens, config.hidden))
        hidden /= np.sqrt(np.mean(hidden ** 2, axis=-1, keepdims=True))
    packed = pack_inputs(hidden, config)
    graph = CompactGraph(config.slots, prepared_qk=args.prepared_qk, prepared_av=args.prepared_av,
                         qk_pack_inputs=args.qk_pack_inputs, av_pack_inputs=args.av_pack_inputs)
    inputs = [graph.input('hidden_' + str(i)) for i in range(config.hidden_tiles)]
    if args.scope == 'encoder':
        outputs = build_model(graph, inputs, weights, config, boot_policy='stage')
    else:
        outputs = graph.compact_linear(inputs, weights[0]['ffn_up_weight'], weights[0]['ffn_up_bias'],
                                       config=config, label='layer0.ffn_up_weight')
    rewrite_stats = None
    if args.algebra_rewrite:
        from equivalent_rewrite import rewrite_compact_graph
        graph, outputs, rewrite_stats = rewrite_compact_graph(graph, outputs, tokens=config.tokens)
    polynomial_stats = None
    if args.polynomial_reassociation:
        from polynomial_rewrite import reassociate_polynomials
        graph, outputs, polynomial_stats = reassociate_polynomials(graph, outputs)
    print('GRAPH', len(graph.nodes), 'constants', len(graph.constants), flush=True)
    lowerer = MultiPrimeLowering(graph, context, profile, initial_level=args.input_level,
                                 strategy=args.strategy,
                                 normalize_shared_rotations=args.shared_rotation_rescale)
    schedule = lowerer.export(outputs, args.output_dir, resident=True)
    reference = dense_reference(hidden, weights, config, args.scope)
    expected = pack_inputs(reference, config, width=reference.shape[1])
    plan = json.loads((args.output_dir / 'model.runtime-plan.json').read_bytes())
    manifest = json.loads((args.output_dir / 'compact-linear.json').read_bytes())
    fixture = dict(format_version=1, context=context, native_boot=native,
        compact_linear_manifest='compact-linear.json', compact_periodic_encoding=args.periodic_encoding,
        prepared_rotation_steps=manifest['rotation_steps'], prepared_direct_rotation_keys=True,
        inputs=[dict(value_id=int(ident), values=row.tolist())
                for ident, row in zip(plan['external_inputs'], packed)],
        expected_outputs=[dict(value_id=int(ident), values=row.tolist())
                          for ident, row in zip(plan['final_outputs'], expected)],
        model=dict(kind='bert-base-encoder', layout='cerium-feature-tiled', config=asdict(config),
            scope=args.scope, seed=args.seed,
            input_sha256=digest(hidden.astype('<f8').tobytes()),
            weight_sha256={str(layer): {key: digest(value.tobytes()) for key, value in w.items()}
                           for layer, w in enumerate(weights)},
            comparison_caveat='Our original polynomial workload; not Cerium nonlinear approximations, paired complex Boot, dynamic-P or exact DFT implementation'))
    if data_binding is not None:
        fixture['model']['data_binding'] = data_binding
        schedule['data_binding'] = data_binding
    if rewrite_stats is not None:
        fixture['model']['algebra_rewrite'] = rewrite_stats
        schedule['algebra_rewrite'] = rewrite_stats
    if polynomial_stats is not None:
        fixture['model']['polynomial_reassociation'] = polynomial_stats
        schedule['polynomial_reassociation'] = polynomial_stats
    if args.shared_rotation_rescale:
        fixture['model']['shared_rotation_rescale'] = True
    if args.prepared_qk:
        qk_manifest = json.loads((args.output_dir / 'prepared-qk.json').read_bytes())
        fixture['prepared_qk_manifest'] = 'prepared-qk.json'
        fixture['prepared_qk_manifest_sha256'] = digest((args.output_dir / 'prepared-qk.json').read_bytes())
        fixture['prepared_rotation_steps'] = sorted(set(fixture['prepared_rotation_steps']) |
                                                   set(qk_manifest['rotation_steps']))
        fixture['model']['prepared_qk'] = True
        if args.qk_pack_inputs:
            fixture['model']['qk_pack_inputs'] = True
            fixture['prepared_qk_required_capabilities'] = ['pack_inputs']
    if args.prepared_av:
        av_manifest = json.loads((args.output_dir / 'prepared-av.json').read_bytes())
        fixture['prepared_av_manifest'] = 'prepared-av.json'
        fixture['prepared_av_manifest_sha256'] = digest((args.output_dir / 'prepared-av.json').read_bytes())
        fixture['prepared_rotation_steps'] = sorted(set(fixture['prepared_rotation_steps']) |
                                                   set(av_manifest['rotation_steps']))
        fixture['model']['prepared_av'] = True
        if args.av_pack_inputs:
            fixture['model']['av_pack_inputs'] = True
            fixture['prepared_av_required_capabilities'] = ['packed_input2_partial']
    (args.output_dir / 'fixture.json').write_bytes(json_bytes(fixture))
    schedule.update(generation_seconds=time.monotonic() - start,
        native_profile_verified=probe is not None,
        native_profile_probe_sha256=digest(args.profile_report.read_bytes()) if probe else None,
        native_profile_fixture_sha256=(digest(args.profile_fixture.read_bytes())
                                       if args.profile_fixture else None),
        polynomial_workload_preserved=True, attention_layout=args.attention_layout,
        periodic_encoding_requested=args.periodic_encoding)
    (args.output_dir / 'schedule.json').write_bytes(json_bytes(schedule))
    if probe:
        (args.output_dir / 'native-profile-probe.json').write_bytes(args.profile_report.read_bytes())
        if args.profile_fixture:
            (args.output_dir / 'native-profile-fixture.json').write_bytes(args.profile_fixture.read_bytes())
    print(json.dumps(schedule, indent=2), flush=True)


if __name__ == '__main__':
    main()
