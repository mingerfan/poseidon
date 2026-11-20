"""S4: bounded Qwen first-block runs using the original model and toolchain.

The cached Nix launcher is reused directly; no file digest is computed. Original
worker implementations are delegated with their compiler metadata context.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType

from .. import paths
from . import qwen_rule63_block_compile as rule
from .r10_paper_metrics import write_json

MODULE = 'pt_agent.experiments.s4_qwen_block'
ROUND = 'S4-QWEN-BLOCK'
DEADLINE = datetime(2026, 11, 18, 12, tzinfo=timezone.utc).timestamp()
GIB = 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def clone(function, replacements):
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    copied = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied


def remaining_before_deadline(budget, *, now=None):
    value = min(budget, int(DEADLINE - (time.time() if now is None else now)))
    if value <= 0:
        raise TimeoutError('S4 publication deadline reached')
    return value


def heldout_decision(seconds_per_sample, available):
    if seconds_per_sample <= 0 or available <= 0:
        raise ValueError('A positive measured Mock sample time and real split are required')
    requested = 256 if seconds_per_sample * 1000 > 7200 else 1000
    # Never invent or repeat samples to achieve a requested sample count.
    return dict(mock_seconds_per_sample=seconds_per_sample,
                estimated_single_1000_sample_pass_s=seconds_per_sample * 1000,
                requested_heldout_samples=requested, actual_heldout_samples=min(requested, available),
                available_original_heldout_samples=available,
                original_split_preserved=True, decision_before_Q1=True,
                split_shortfall=available < requested)


def enter_cached(argv):
    """Use the same pure, offline Nix shell without the wrapper's digest check."""
    scripts = paths.REPO / 'DSL/DSL_agent/scripts'
    sys.path.insert(0, str(scripts / 'baseline'))
    import hecate_python_env as environment
    from ..tools import nixrun
    runtime = paths.PLATFORM_DIR / 'deps/nix-portable-v012'
    launcher = runtime / 'nix-portable'
    if not launcher.is_file():
        raise FileNotFoundError(str(launcher))
    old = environment.WRAPPER
    previous_argv = sys.argv
    launcher_env = dict(NP_LOCATION=str(runtime), NP_GIT='/usr/bin/git', NP_RUNTIME='bwrap',
        NP_BWRAP='/usr/bin/bwrap', XDG_CACHE_HOME=str(paths.PLATFORM_DIR / 'cache/nix-portable-v012'),
        XDG_CONFIG_HOME=str(runtime / 'xdg-config'), XDG_STATE_HOME=str(runtime / 'xdg-state'),
        TMPDIR=str(paths.PLATFORM_DIR / 'cache/nix-portable-tmp'))
    environment.WRAPPER = ['env', '-u', 'NP_DEBUG', '-u', 'NP_RUN',
        *[key + '=' + value for key, value in launcher_env.items()],
        'flock', '--exclusive', '--timeout', '30', str(runtime / 'launcher.lock'), str(launcher)]
    try:
        sys.argv = [str(nixrun.__file__), *argv]
        return nixrun.main()
    finally:
        sys.argv = previous_argv
        environment.WRAPPER = old


def phase_command(args, phase, remaining):
    seconds = min(7200 if phase == 'compile' else 1800,
                  remaining_before_deadline(max(1, int(remaining))) - 45)
    if seconds <= 0:
        raise TimeoutError('S4 Q0 inclusive 1800-second budget exhausted')
    inner = ['-B', '-m', MODULE, '--q0-worker', '--worker-phase', phase,
        '--out', str(args.out), '--work-dir', str(args.out / 'work' / phase),
        '--phase-seconds', str(seconds), '--case-gib', str(args.case_gib)]
    for name in rule.SOURCE_OPTIONS:
        inner += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    return [sys.executable, '-B', '-m', MODULE, '--enter', '--seconds', str(seconds + 10),
            '--python', str(paths.INTAKE_PYTHON), '--', *inner], seconds + 25


def q0_record(args, result, completion):
    row = rule.paper_record(args, result, completion)
    row.update(task='S4-Q0', budget_s=1800, publication_deadline_utc='2026-11-18T12:00:00+00:00',
               paper_eligible=time.time() <= DEADLINE,
               native_compiler_version_unchanged=True, extra_integrity_checks=0)
    if result.get('incoming_signal') == 14:
        row.update(status='timeout', status_detail='Q0 inclusive1800-second budget exhausted')
    return row


def q0(argv):
    args = rule.arguments(argv)
    if args.seconds > 1800:
        raise ValueError('S4 Q0 is limited to1800seconds inclusive')
    args.seconds = remaining_before_deadline(args.seconds)
    execute = clone(rule.execute, dict(phase_command=phase_command, paper_record=q0_record,
                                     ROUND='S4-Q0-QWEN-RULE63-R0-BLOCK'))
    with (paths.RESULTS_DIR / 'autonomous-heavy-experiments.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute(args)


def q0_worker(argv):
    args = rule.arguments(argv)
    from .r29_local_compile_matrix import native_metadata_commands
    with native_metadata_commands(read(args.metadata_template)):
        return rule.worker(args)


def probe(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--metadata-template', type=Path, required=True)
    parser.add_argument('--model-info', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--source-program', type=Path, required=True)
    args = parser.parse_args(argv)
    from .r29_local_compile_matrix import native_metadata_commands
    with native_metadata_commands(read(args.metadata_template)):
        import numpy as np
        from dataclasses import replace
        from transformers import AutoTokenizer
        from ..backends.streaming_mock import StreamingMockBackend
        from ..dsl.parser import parse
        from ..kernels.compact_linear import compact_transformer_projections
        from ..lower import lower, pack_inputs, decode_outputs
        from ..models.mapped_info import load_mapped_info
        from .qwen_layered_structure import split_program
        from .qwen_agent_continuation import TokenSplits
        info = load_mapped_info(args.model_info)
        program = parse(args.source_program.read_text())
        block = split_program(program, {key: value.shape for key, value in info.graph.weights.items()}).layers[0].program
        # Segmentation exposes the two unchanged mask handles as linkage
        # outputs. A standalone block benchmark returns only its computed hidden state.
        block = replace(block, outputs=block.outputs[:1])
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
        access = []
        arrays, _ = TokenSplits(info, args.inputs, tokenizer, access).get('heldout', 1)
        backend = StreamingMockBackend(16384, batch=1)
        began = time.perf_counter_ns()
        try:
            backend.bind(pack_inputs(block, [arrays[p.name] for p in block.params]))
            with compact_transformer_projections():
                outputs, layouts = lower(block, info.graph.weights, backend, act_mode='program')
            result = decode_outputs([value.value for value in outputs], layouts)[0]
            elapsed = (time.perf_counter_ns() - began) / 1e9
            if result.shape != (1, 32, 896) or not np.isfinite(result).all():
                raise ValueError('Qwen diagnostic packed probe shape or finite-value check failed')
        finally:
            backend.release_batch()
        value = dict(status='ok', error=None, mock_seconds_per_sample=elapsed,
            samples=1, sample_accesses=access, source_program=str(args.source_program),
            source_use='Existing unfrozen candidate block for simulation runtime only; Q0 R0 failed before a program existed',
            placement_replayed=False, candidate_warm_start=False, report_samples=0, confirm_samples=0,
            quality_or_native_compilation_claimed=False,
            heldout_decision=heldout_decision(elapsed, 1000))
        write_json(args.out, value)
        return value


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--enter', action='store_true')
    parser.add_argument('--q0-worker', action='store_true')
    parser.add_argument('--q0', action='store_true')
    parser.add_argument('--probe', action='store_true')
    selected, remaining = parser.parse_known_args(argv)
    if selected.enter:
        return enter_cached(remaining)
    if selected.q0_worker:
        result = q0_worker(remaining)
    elif selected.q0:
        result = q0(remaining)
    elif selected.probe:
        result = probe(remaining)
    else:
        parser.error('Select one S4 operation')
    print(json.dumps(dict(status=result['status'], error=result.get('error'))))
    return int(result['status'] != 'ok')


if __name__ == '__main__':
    raise SystemExit(main())
