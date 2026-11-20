"""Original BERT/SST-2 intake, resumable internal Agent steps and frozen report.

Every path is supplied by the outer process.  This module never loads the run
registry, reads the hidden-label test shard, or executes generated source.
"""

import argparse
from contextlib import nullcontext
import fcntl
import json
import os
from pathlib import Path
import sys

from .r10_paper_metrics import StageRecorder, write_json


PROMPT_VERSION = 'r8-v11'
TUNE_RULE = 'v8-6.25'
PLACEMENT = 'program-plan'
FORMAL = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency', 'accuracy_acceptance')


def initialize_numeric():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import torch
    if not getattr(initialize_numeric, 'done', False):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        initialize_numeric.done = True


def strict_math(reference, observed):
    import numpy as np

    reference, observed = np.asarray(reference), np.asarray(observed)
    if reference.dtype != np.float64 or observed.dtype != np.float64 or reference.shape != observed.shape:
        raise ValueError('Independent BERT mathematics requires equal shapes and float64 values')
    if not reference.size or not np.isfinite(reference).all() or not np.isfinite(observed).all():
        raise ValueError('Independent BERT mathematics requires nonempty finite outputs')
    difference = np.abs(reference - observed)
    tolerance = 1e-12 + 1e-12 * np.abs(reference)
    return dict(passed=bool(np.all(difference <= tolerance)),
                max_absolute_error=float(difference.max()),
                max_tolerance_ratio=float(np.max(difference / tolerance)),
                criterion='abs(observed-reference) <= 1e-12 + 1e-12*abs(reference)')


def _initial_result(phase):
    return dict(status='running', phase=phase, model='bert-tiny-sst2',
                import_completed=False, frozen=False, report_completed=False,
                native_test_reads=0, report_samples=0, llm_calls=0,
                compiler_runs=0, keygen_runs=0, real_ckks_attempts=0, real_ckks_completed=0,
                **{name: None for name in FORMAL})


def prepare(args, recorder, result):
    import numpy as np
    import torch
    from ..analyzer import calibrate, reference_forward
    from ..data.sst2_source import SST2Samples
    from ..intake.hf_frontend import boundary_module
    from ..intake.probe import import_model

    if args.model_dir.exists():
        raise FileExistsError('A prepared BERT import is retained: ' + str(args.model_dir))
    with recorder.span('sst2_source_metadata'):
        raw = SST2Samples(args.inputs, data_python=args.data_python)
        result['source'] = raw.record()
        result['split_counts'] = {name: len(raw.partition[name]) for name in ('calibration', 'heldout', 'confirm', 'report')}
    with recorder.span('import_model'):
        imported = import_model(args.model, raw, observer=recorder)
    info = imported.info
    cfg = imported.original.config
    if (cfg.model_type != 'bert' or info.task != 'classification'
            or info.metric != 'classification_top1' or info.threshold != {'acc_drop_pp': 0.5}
            or getattr(cfg, 'num_labels', None) != 2):
        raise ValueError('This entry requires the supplied original binary-classification BERT model')
    info.source.update(inputs_path=str(args.inputs.resolve()), sst2=raw.record())
    result['model_metadata'] = dict(model_type=cfg.model_type, architectures=cfg.architectures,
        num_hidden_layers=cfg.num_hidden_layers, hidden_size=cfg.hidden_size,
        num_labels=cfg.num_labels, unchanged_original_model=str(args.model.resolve()))
    if getattr(info.graph, 'lowering_errors', ()):
        raise ValueError('Original BERT import retained unsupported operations')
    with recorder.span('normalized_reference_checks'):
        wrapper = boundary_module(imported.original)
        result['reference_checks'] = []
        for index in info.split['calibration'][1:4]:
            arrays, _ = imported.samples.select([index])
            with torch.no_grad():
                expected = wrapper(**{name: torch.from_numpy(value) for name, value in arrays.items()}).cpu().numpy()
            observed = reference_forward(info.graph, arrays)[info.graph.outputs[0]]
            check = dict(sample_id=raw.ids[index], **strict_math(expected, observed))
            result['reference_checks'].append(check)
            if not check['passed']:
                raise ValueError('Normalized original BERT differs from the supplied model')
    with recorder.span('save_model_info'):
        info.save(args.model_dir)
    samples_dir = args.model_dir / 'samples'
    samples_dir.mkdir()
    write_json(samples_dir / 'source.json', raw.record())
    for split in ('calibration', 'heldout'):
        with recorder.span('prepare_' + split):
            arrays, labels = imported.samples.get(split)
            if labels is None or len(labels) != len(info.split[split]):
                raise ValueError('SST-2 internal labels are incomplete')
            np.savez(samples_dir / (split + '.npz'), **arrays, labels=labels)
            if split == 'calibration':
                with recorder.span('calibrate'):
                    calibration = calibrate(info.graph, arrays, batch=20)
                write_json(args.model_dir / 'calibration.json', calibration)
    result.update(import_completed=True, model_dir=str(args.model_dir), sequence=info.sequence,
                  input_handling=info.input_handling, sample_accesses=raw.accesses,
                  training_samples_read=len(raw._cache), confirm_samples=0,
                  task=info.task, metric=info.metric, task_policy=info.threshold)


class NumericSplits:
    """Materialize embeddings by split, with an explicit frozen report gate."""
    def __init__(self, model_dir, info, *, frozen=None, program=None):
        from ..data.sst2_source import SST2Samples
        self.directory, self.info = Path(model_dir) / 'samples', info
        self.raw = SST2Samples(info.source['inputs_path'], record=info.source['sst2'])
        if self.raw.split() != info.split:
            raise ValueError('Saved BERT split and original SST-2 source differ')
        if json.loads((self.directory / 'source.json').read_text()) != self.raw.record():
            raise ValueError('Stored BERT embeddings and SST-2 source differ')
        self.local = None
        if frozen is not None and program is not None:
            self.raw.authorize_report(frozen, program)

    def get(self, split, n=None):
        import numpy as np
        from ..intake.hf_frontend import HFInputs, load_local
        if split not in ('calibration', 'heldout', 'confirm', 'report'):
            raise ValueError('Unknown SST-2 sample split')
        if split == 'report' and not self.raw.report_authorized:
            raise ValueError('SST-2 report requires the frozen access capability')
        path = self.directory / (split + '.npz')
        if path.exists() and split != 'report':
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name][:n]) for name in self.info.graph.inputs}
                labels = np.array(archive['labels'][:n])
            if any(value.dtype != np.float64 for value in arrays.values()) or labels.dtype != np.int64:
                raise ValueError('Stored BERT inputs must retain float64 and native integer labels')
            expected = len(self.info.split[split]) if n is None else min(n, len(self.info.split[split]))
            if len(labels) != expected or any(len(value) != expected for value in arrays.values()):
                raise ValueError('Stored BERT input count differs from the fixed partition')
            return arrays, labels
        if self.local is None:
            self.local = load_local(self.info.source['local_path'], attn_implementation='eager')
        arrays, labels = HFInputs(self.local, self.raw, self.info).get(split, n)
        if split != 'report' and n is None:
            np.savez(path, **arrays, labels=labels)
        return arrays, labels


def load_context(args, *, frozen=False):
    from ..models.info import ModelInfo
    from ..planners.agent_tools import build_context
    info = ModelInfo.load(args.model_dir)
    if (Path(info.source['local_path']).resolve() != args.model.resolve()
            or Path(info.source['inputs_path']).resolve() != args.inputs.resolve()):
        raise ValueError('BERT session must retain its original model and input source')
    samples = NumericSplits(args.model_dir, info,
        **(dict(frozen=args.session_dir / 'FROZEN.json', program=args.session_dir / 'program.ptdsl') if frozen else {}))
    ctx = build_context(info, json.loads((args.model_dir / 'calibration.json').read_text()),
                        samples=samples, n_val=4 if frozen else 1000)
    ctx.prompt_version, ctx.placement, ctx.tune_search = PROMPT_VERSION, PLACEMENT, 'sweep'
    return ctx


class RecordedRenewingClient:
    """Retain exact requests and responses while the global allowance renews."""
    def __init__(self, client, directory, recorder=None):
        self.client, self.directory, self.recorder = client, Path(directory), recorder
        self.directory.mkdir(parents=True, exist_ok=True)
        self.calls = 0

    def chat(self, messages, *, purpose):
        from ..llm import LLMError
        with (self.directory / 'client.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            number = len(list(self.directory.glob('call-*.request.json'))) + 1
            stem = 'call-' + str(number).zfill(4)
            write_json(self.directory / (stem + '.request.json'), dict(messages=messages, purpose=purpose))
            self.calls += 1
            span = self.recorder.span('llm_call', purpose=purpose, session_call=number) if self.recorder else nullcontext()
            try:
                with span:
                    reply = self.client.chat(messages, purpose=purpose)
            except LLMError as exc:
                write_json(self.directory / (stem + '.error.json'), dict(error=exc.code, diagnostics=exc.diagnostics))
                raise
            write_json(self.directory / (stem + '.response.json'), reply)
            return reply


def prompt(args, recorder, result):
    from ..autonomous_budget import audit_public_messages
    from ..llm import Config
    from ..planners.prompts import build_system_prompt, initial_message
    from ..planners.transformer_session import load_state, output_tokens
    with recorder.span('build_internal_context'):
        ctx = load_context(args)
    state_path = args.session_dir / 'SESSION.json'
    state = load_state(state_path) if state_path.exists() else None
    phase = state['phase'] if state is not None else 'evaluate'
    if phase != 'evaluate':
        result.update(status='ok', next_step=phase, api_required=False)
        return
    messages = state['messages'] if state is not None else [dict(role='system', content=build_system_prompt(ctx)),
                                                          dict(role='user', content=initial_message(ctx))]
    payload = Config(model='deepseek-flash', reasoning_effort='high',
                     max_tokens=output_tokens(PROMPT_VERSION), timeout_seconds=600).settings()
    payload['messages'] = messages
    audit = audit_public_messages(messages)
    destination = args.work_dir / 'public-api-request.json'
    write_json(destination, dict(endpoint='https://api.deepseek.com/chat/completions', payload=payload,
        provenance='Original public BERT architecture and internal training calibration/heldout aggregates',
        raw_sentences_included=False, raw_weights_included=False, report_samples=0,
        message_audit=audit))
    result.update(status='ok', api_required=True, request=str(destination), next_step=phase)


def step(args, recorder, result):
    from ..autonomous_budget import RenewingLLMClient
    from ..planners.transformer_session import load_state, output_tokens, session_step
    with recorder.span('build_internal_context'):
        ctx = load_context(args)
    state_path = args.session_dir / 'SESSION.json'
    state = load_state(state_path) if state_path.exists() else None
    phase = state['phase'] if state is not None else 'evaluate'
    if phase == 'done':
        result.update(status='ok', next_step='done', frozen=(args.session_dir / 'FROZEN.json').exists())
        return
    client = None
    if phase == 'evaluate':
        client = RecordedRenewingClient(RenewingLLMClient(budget_path=args.ledger,
            allowance_path=args.allowance, recorder=recorder, artifact_dir=args.session_dir / 'api', model='deepseek-flash',
            reasoning_effort='high', max_tokens=output_tokens(PROMPT_VERSION), timeout=600),
            args.session_dir / 'calls', recorder=recorder)
    with recorder.span('agent_' + phase):
        measured = session_step(ctx, args.session_dir, client=client, max_iters=10,
            tune_rule=TUNE_RULE, confirm_rule='c2', tune_policy='all', stale_limit=2,
            focus_retune=True, numeric_lock=True)
    result.update(status='ok', step=measured, next_step=measured['next_step'],
                  llm_calls=client.calls if client else 0,
                  frozen=(args.session_dir / 'FROZEN.json').exists(), report_samples=0)


def report(args, recorder, result):
    import numpy as np
    from dataclasses import replace
    from ..analyzer import reference_forward
    from ..data.sst2_source import require_frozen
    from ..dsl.interface import require_interface
    from ..dsl.parser import parse
    from ..metrics import passes
    from ..transformer_validation import execute, validate

    marker, source = args.session_dir / 'FROZEN.json', args.session_dir / 'program.ptdsl'
    require_frozen(marker, source)
    access = args.session_dir / 'REPORT_ACCESS.json'
    with access.open('x') as stream:
        json.dump(dict(program=str(source), stage='checking-frozen-program', complete=False), stream)
    with recorder.span('frozen_context'):
        ctx = load_context(args, frozen=True)
        program = parse(source.read_text())
        require_interface(program, ctx.model_info)
    with recorder.span('frozen_mathematical_checks'):
        arrays, _ = ctx.samples.get('calibration', 3)
        result['frozen_math'] = []
        for index in range(3):
            inputs = {name: value[index:index+1] for name, value in arrays.items()}
            exact = reference_forward(ctx.graph, inputs)[ctx.graph.outputs[0]]
            tensor_exact = execute(program, ctx.weights, inputs, engine='tensor', exact=True)
            mock_exact = execute(program, ctx.weights, inputs, engine='mock', exact=True)
            tensor_program = execute(program, ctx.weights, inputs, engine='tensor')
            mock_program = execute(program, ctx.weights, inputs, engine='mock')
            checks = [strict_math(exact, tensor_exact), strict_math(exact, mock_exact),
                      strict_math(tensor_program, mock_program)]
            result['frozen_math'].append(dict(sample=index, checks=checks))
            if not all(check['passed'] for check in checks):
                raise ValueError('Frozen BERT independent engine mathematics failed')
    with recorder.span('read_frozen_report'):
        inputs, labels = ctx.samples.get('report')
    if len(labels) != 872:
        raise ValueError('SST-2 frozen report requires all 872 native validation samples')
    with recorder.span('report_reference'):
        exact = []
        for start in range(0, len(labels), 20):
            values = reference_forward(ctx.graph, {name: value[start:start+20] for name, value in inputs.items()})
            exact.append(values[ctx.graph.outputs[0]])
        exact = np.concatenate(exact)
    ctx = replace(ctx, val_inputs=inputs, val_images=inputs[ctx.graph.inputs[0]], val_labels=labels,
                  val_exact_logits=exact, val_split='report', val_exact_top1=float(100*np.mean(exact.argmax(1) == labels)))
    with recorder.span('report_tensor_program'):
        captured = []
        measured = validate(program, ctx, engine='tensor', stop_on_divergence=False, output_callback=captured.append)
    if len(captured) != 1 or captured[0].shape != exact.shape:
        raise ValueError('Frozen BERT report outputs are incomplete')
    output = captured[0]
    finite = bool(np.isfinite(output).all())
    result.update(status='ok', frozen=True, report_completed=True, report_samples=len(labels),
                  validation_metrics=measured, task_policy=ctx.model_info.threshold,
                  task_policy_passed=bool(finite and passes(measured, ctx.model_info.threshold)),
                  all_finite=finite, sample_accesses=ctx.samples.raw.accesses)
    np.savez(args.work_dir / 'report-outputs.npz', reference=exact, approximate=output, labels=labels)
    write_json(access, dict(program=str(source), complete=True, report_samples=len(labels),
                            result=str(args.work_dir / 'worker_result.json')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'prompt', 'step', 'report'), required=True)
    for name in ('model', 'inputs', 'model-dir', 'session-dir', 'work-dir', 'ledger', 'allowance', 'data-python'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ('model', 'inputs', 'model_dir', 'session_dir', 'work_dir', 'ledger', 'allowance'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.data_python = args.data_python.expanduser().absolute()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result = _initial_result(args.phase)
    path = args.work_dir / 'worker_result.json'
    recorder = StageRecorder(args.work_dir / 'events.jsonl')
    try:
        initialize_numeric()
        with recorder.span('bert_sst2_' + args.phase):
            globals()[args.phase](args, recorder, result)
        result['status'] = 'ok'
    except Exception as exc:
        result.update(status='error', error=str(type(exc).__name__) + ': ' + str(exc))
    finally:
        from ..planners.transformer_session import json_safe
        write_json(path, json_safe(result))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
