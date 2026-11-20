"""Delegate unchanged BERT phases under recorded compiler-command metadata.

The original default entry remains available. This optional entry installs the
command context before importing the original worker or its cost dependencies.
"""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
from types import FunctionType

from .. import paths
from .r10_paper_metrics import write_json
from .r29_local_compile_matrix import compiler_argv, native_metadata_commands

WORKER='pt_agent.experiments.bert_sst2_worker'
OUTER='pt_agent.experiments.bert_sst2'
DEFAULT_STEP_SECONDS=7200
MAX_STEP_SECONDS=21600


def validate_step_seconds(value):
    """A finite integer wall allowance; the original default is unchanged."""
    if type(value)is not int or not DEFAULT_STEP_SECONDS<=value<=MAX_STEP_SECONDS:
        raise ValueError('Step seconds must be an integer within7200..21600')
    return value


def private_outer_namespace(outer,step_seconds):
    """Bind original outer functions to a private phase-clock dictionary only."""
    selected=validate_step_seconds(step_seconds)
    original=vars(outer);namespace=dict(original)
    namespace['PHASE_SECONDS']=dict(outer.PHASE_SECONDS,step=selected)
    for name,value in original.items():
        # contextmanager wrappers keep their original contextlib globals.
        if isinstance(value,FunctionType)and value.__globals__ is original:
            clone=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            clone.__kwdefaults__=value.__kwdefaults__;namespace[name]=clone
    return namespace


def worker_delegate(argv,metadata_template,step_seconds=DEFAULT_STEP_SECONDS):
    step_seconds=validate_step_seconds(step_seconds)
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--work-dir',type=Path,required=True)
    selected,_=parser.parse_known_args(argv);work=selected.work_dir.resolve();work.mkdir(parents=True,exist_ok=True)
    template=json.loads(Path(metadata_template).read_text())
    compiler_argv(template,'pars',binary='/compiler',source='/input',profile='/profile',output='/lowered')
    snapshot=work/'recorded-command-entry.py'
    if snapshot.exists():raise ValueError('Retain the prior optional BERT entry evidence')
    shutil.copyfile(Path(__file__),snapshot);shutil.copyfile(metadata_template,work/'recorded-compile-template.json')
    write_json(work/'recorded-command-delegation.json',dict(wrapper=str(Path(__file__).resolve()),
        original_worker=WORKER,original_argv=list(argv),metadata_template=str(Path(metadata_template).resolve()),
        configured_step_seconds=step_seconds,
        early_context_installed_before_worker_import=True,new_digest_computations=0,
        digest_counter_scope='New operator identifier digest computations in this Python command adapter only',
        builtin_serialization_digest_scope='Native payload/manifest/plan serialization may generate its existing format identifiers; this counter does not count or verify those identifiers',
        scope='Carry existing operator-spec identifier as compiler metadata; no added content or integrity verification. Original BERT phase and Provider loader unchanged.'))
    with native_metadata_commands(template):
        worker=importlib.import_module(WORKER)
        shutil.copyfile(Path(worker.__file__),work/'original-bert-phase-source.py')
        return worker.main(list(argv))


def outer_delegate(argv,metadata_template,step_seconds=DEFAULT_STEP_SECONDS):
    step_seconds=validate_step_seconds(step_seconds)
    outer=importlib.import_module(OUTER);original=outer.command
    if step_seconds!=DEFAULT_STEP_SECONDS:
        namespace=private_outer_namespace(outer,step_seconds);base=namespace['command']
        def private_command(args,phase,work):
            result=base(args,phase,work)
            if result.count(outer.WORKER)!=1:raise ValueError('Original BERT command must name its worker exactly once')
            result[result.index(outer.WORKER)]=__spec__.name
            if phase=='step':result+=['--step-seconds',str(step_seconds)]
            return result+['--delegate-worker','--metadata-template',str(Path(metadata_template).resolve())]
        namespace['command']=private_command
        return namespace['main'](list(argv))
    def command(args,phase,work):
        result=original(args,phase,work)
        if result.count(outer.WORKER)!=1:raise ValueError('Original BERT command must name its worker exactly once')
        result[result.index(outer.WORKER)]=__spec__.name
        return result+['--delegate-worker','--metadata-template',str(Path(metadata_template).resolve())]
    outer.command=command
    try:return outer.main(list(argv))
    finally:outer.command=original


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--delegate-worker',action='store_true')
    parser.add_argument('--step-seconds',type=int,default=DEFAULT_STEP_SECONDS)
    parser.add_argument('--metadata-template',type=Path,
        default=paths.RESULTS_DIR/'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json')
    args,remaining=parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    try:validate_step_seconds(args.step_seconds)
    except ValueError as exc:parser.error(str(exc))
    if args.delegate_worker:
        return worker_delegate(remaining,args.metadata_template.resolve(),args.step_seconds)
    return outer_delegate(remaining,args.metadata_template.resolve(),args.step_seconds)


if __name__=='__main__':raise SystemExit(main())
