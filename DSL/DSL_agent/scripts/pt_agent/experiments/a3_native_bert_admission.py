"""A3: record native semantic-entry limitations and the own HEIR adapter gap."""
import argparse
import ast
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

from .. import paths
from .a_group_runs import empty_record, append_record
from .r10_paper_metrics import write_json
from .completion_metrics import artifact_bytes, persist_case_completion


def closed_arithmetic_tables(path):
    tree = ast.parse(Path(path).read_text())
    tables = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('toUnary', 'toBinary', 'toInnerUnary'):
                    tables[target.id] = dict(line=node.lineno, names=sorted(ast.literal_eval(node.value)))
    if set(tables) != {'toUnary', 'toBinary', 'toInnerUnary'}:
        raise ValueError('Native arithmetic registry changed; reassess the source evidence')
    if any(set(v['names']) & {'softmax', 'layernorm', 'gelu'} for v in tables.values()):
        raise ValueError('Native semantic registry changed; assess actual model support')
    return tables


def execute(args):
    started = time.perf_counter_ns()
    if args.out.exists() or args.out.parent != paths.RESULTS_DIR.resolve():
        raise ValueError('Use a fresh direct RESULTS directory')
    evidence = json.loads(args.evidence.read_text())
    if evidence.get('activity', {}).get('library_modifications') != 0:
        raise ValueError('Require a read-only capability review')
    args.out.mkdir()
    completed = []
    for system, group, expr in (
        ('dacapo_native', 'dacapo-native', paths.DACAPO_UPSTREAM/'python/hecate/hecate/expr.py'),
        ('orbit', 'orbit-native', paths.ORBIT_ROOT/'frontend/dacapo_patch/python/hecate/expr.py'),
        ('heir', 'heir-native', None),
    ):
        decision = evidence['decisions'][system]
        detail = dict(system=system, source_evidence=str(args.evidence), compiler_attempts=0,
                      model_forwards=0, sample_reads=0, API=0, CKKS=0, extra_integrity_checks=0)
        if expr is not None:
            if decision['recommendation'] != 'unsupported':
                raise ValueError('Original native support assessment changed')
            detail.update(status='unsupported', semantic_operators=['softmax', 'layernorm', 'gelu'],
                          registry_path=str(expr), tables=closed_arithmetic_tables(expr),
                          evidence_locations=decision['evidence_locations'])
            reason = ('Unmodified native semantic frontend lacks softmax, LayerNorm and GELU lowering. '
                      'Polynomial decomposition by an expert remains possible. Source: ' +
                      '; '.join(decision['evidence_locations']))
        else:
            from .heir_cnn_export import saved_activations
            comparison = json.loads((args.bert_case/'result.json').read_text())
            if comparison.get('status') != 'ok' or comparison.get('compile_accepted') is not True:
                raise ValueError('Require the closed current BERT fixed-rule program')
            try:
                saved_activations((args.bert_case/'work/rule63.ptdsl').read_text())
            except ValueError as exc:
                detail.update(status='driver_error', own_adapter_admission_attempts=1,
                              own_adapter_exception=str(exc),
                              own_adapter='pt_agent.experiments.heir_cnn_export.saved_activations',
                              official_softmax_and_GELU_paths_exist=True,
                              baseline_unsupported_proven=False,
                              source_program=str(args.bert_case/'work/rule63.ptdsl'))
            else:
                raise ValueError('Own HEIR adapter now admits BERT; schedule actual native compilation')
            reason = ('Own HEIR adapter rejects the current BERT GELU/tanh program before compiler launch: '
                      + detail['own_adapter_exception'] + '. Official same-release softmax/GELU paths exist; '
                      'full BERT lowering is not connected. Source: heir_cnn_export.py:saved_activations/export_graph.')
        destination = args.out/(system+'.json')
        write_json(destination, detail)
        row = empty_record(run_id=args.out.name+'-'+system, task='A3', model='bert_tiny',
                           system=system, param_group=group, status=detail['status'],
                           status_detail=reason, budget_s=7200, api_calls=0,
                           source=str(destination), finished_utc=datetime.now(timezone.utc).isoformat(),
                           compiler_attempts=0, source_evidence=str(args.evidence))
        append_record(args.paper_runs, row)
        completed.append(dict(system=system, status=detail['status'], source=str(destination)))
    write_json(args.out/'result.json', dict(status='ok', task='A3', rows=completed,
                                          compiler_attempts=0, actual_native_BERT_compilation_completed=False))
    from ..registry import git_rev
    registry = dict(run_id=args.out.name, round='A3-BERT-NATIVE-ADMISSION', model='bert_tiny',
                    mode='native-semantic-source-and-own-adapter-admission', git_rev=git_rev(),
                    metrics=dict(status='ok', rows=completed, compiler_attempts=0, API=0, CKKS=0),
                    artifacts_dir=str(args.out), ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(registry, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    persist_case_completion(args.out, round_label='A3-BERT-NATIVE-ADMISSION', started_ns=started,
                            status='ok', timing=dict(status='ok', attribution_ready=False,
                                                    spans=[], diagnostics=[]),
                            required_artifact_bytes=artifact_bytes(args.out))
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('out', 'evidence', 'bert-case', 'paper-runs'):
        parser.add_argument('--'+key, required=True, type=Path)
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        setattr(args, key, value.expanduser().resolve())
    print(json.dumps(execute(args)))


if __name__ == '__main__':
    main()
