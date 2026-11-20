"""Original VGG math/native pipeline using the existing R16a host runtime.

Keep the identified BERT/batch guard and original initializer lock. Change only
the numeric-worker launch environment, with a separate runtime/timing identity.
The original strict three-training-image check still precedes native lowering.
"""
import argparse
import ast
import copy
import inspect
import os
from pathlib import Path
import shutil
import textwrap
from types import SimpleNamespace

from .. import paths
from . import heir_vgg_cpu_overlap as base

original=base.original
overlap=base.overlap
ROUND='HEIR-VGG-HOST-CPU-OVERLAP'
MODE='original-fused-vgg-training-math-native-existing-host-runtime-during-bert-tune4'
MODULE='pt_agent.experiments.heir_vgg_host_cpu_overlap'
OUT_PREFIX='heir-vgg16-fused-pool-conv-host-cpu-overlap-'
GIB=base.GIB
OBSERVED_HOST_VERSIONS=dict(python='3.11.11',numpy='2.1.3',torch='2.5.1',torch_threads=1)
SCOPE=(base.SCOPE.replace('BERT tune3','BERT tune4')+' Existing R16a Orion host interpreter: Python3.11.11/NumPy2.1.3/Torch2.5.1; '
       'runtime differs from the original pure Nix entry. Clock is CPU-overlap plus host-runtime variant.')
REMOVE_PREFIXES=('LD_','NIX_','__NIX_','HECATE_')
REMOVE_KEYS=frozenset(('IN_NIX_SHELL','PYTHONHOME','PYTHONPATH','PYTHONSTARTUP','PYTHONUSERBASE',
    'PYTHONPLATLIBDIR','PYTHONSAFEPATH','VIRTUAL_ENV','CONDA_PREFIX','TORCH_USE_RTLD_GLOBAL','TORCH_COMPILE_STROBELIGHT'))
THREAD_KEYS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
IDENTIFIED_BERT=(93984,2098305)
IDENTIFIED_BATCH=(36762,961036)


def metadata_admission(args):
    """Exact observed tune4 counterpart of the original metadata predicates.

    Keep raw files and the original shared guard unchanged. This private helper
    admits only the current recorded worker and step, with all original path,
    active-result, original-supervisor and queue predicates still required.
    """
    if ((args.bert_pid,args.bert_starttime)!=IDENTIFIED_BERT
            or(args.batch_pid,args.batch_starttime)!=IDENTIFIED_BATCH):
        raise ValueError('Preserve the exact observed tune4 BERT and original batch identities')
    state=overlap.read(args.bert_case/'session/SESSION.json')
    if (state.get('phase')!='tune'or type(state.get('iteration'))is not int
            or state['iteration']!=4 or state.get('failed')is not None
            or type(state.get('calls'))is not int or state['calls']!=4):
        raise ValueError('The original BERT session must still be exact tune4/calls4 with no failure')
    run=overlap.read(args.bert_run_case/'result.json')
    if (run.get('round')!='BERT-SERIAL-CONTINUATION'or run.get('model')!='bert-tiny-sst2'
            or run.get('status')!='running'or run.get('termination')is not None
            or not overlap._same((run.get('source')or{}).get('source_case'),args.bert_case)
            or(run.get('attempted')or{}).get('step')!=3
            or(run.get('completed')or{}).get('step')!=2):
        raise ValueError('The identified original BERT continuation must retain step3 attempted/step2 completed')
    supervisor=overlap.read(args.batch_case/'supervisor.json');process=supervisor.get('process')or{}
    if(supervisor.get('pid')!=args.batch_pid or process.get('pid')!=args.batch_pid
            or process.get('starttime')!=args.batch_starttime):
        raise ValueError('Saved original batch supervisor identity differs')
    active=overlap.read(args.batch_case/'active.json');argv=active.get('argv')or[]
    if(active.get('status')!='running'or not overlap._same(active.get('out'),args.bert_run_case)
            or argv.count('pt_agent.experiments.bert_serial_continuation')!=1
            or argv.count('--source-case')!=1
            or not overlap._same(argv[argv.index('--source-case')+1],args.bert_case)):
        raise ValueError('Original batch must still own the identified BERT continuation')
    if(args.batch_case/'result.json').exists():
        raise ValueError('Original heavy batch has already written its terminal result')
    return dict(phase='tune',iteration=4,calls=4,failed=None,
        bert_case=str(args.bert_case),bert_run_case=str(args.bert_run_case),
        batch_case=str(args.batch_case),batch_job=active.get('id'))


def host_python():return (paths.ORION_ROOT.parent/'orion-venv/bin/python').absolute()


def host_environment(env=None):
    child={k:v for k,v in (os.environ if env is None else env).items()
        if k not in REMOVE_KEYS and not k.startswith(REMOVE_PREFIXES)}
    child.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',**dict.fromkeys(THREAD_KEYS,'1'))
    return child


def worker_command(source,out,interpreter,memory_gib=24):
    if original.validate_memory_gib(memory_gib)!=24 or Path(interpreter).absolute()!=host_python():
        raise ValueError('Preserve the fixed existing R16a host interpreter and AS24 worker')
    command=[str(interpreter),'-B','-m',original.MODULE,'--worker','--out',str(out),'--memory-gib','24']
    for name in original.SOURCE_FIELDS:command+=['--'+name.replace('_','-'),str(source[name])]
    return command


def runtime_record(args):
    return dict(variant='r16a-existing-orion-host-runtime',interpreter=str(args.python),
        last_observed_import_versions=OBSERVED_HOST_VERSIONS.copy(),
        version_observation_scope='One isolated120second/AS4GiB imports-only check before implementation; no repeated version subprocess in the real worker',
        original_Nix_interpreter=str(paths.INTAKE_PYTHON),original_Nix_versions_observed_here=False,
        original_Nix_runtime_equivalence_claimed=False,original_worker_code_unchanged=True,
        text_or_sample_or_weight_reads_in_import_check=0,nix_launcher_used=False,
        numerical_threads=1,strict_reference_or_tolerance_changed=False)


def _namespace(args):
    # Clone ordinary own wrapper functions only. Imported contextmanagers and
    # all original numerical/native functions retain their own globals/code.
    guard_namespace=dict(vars(overlap),_metadata_admission=metadata_admission)
    guard=overlap._clone_plain(overlap.parallel_observation,guard_namespace)
    private_overlap=SimpleNamespace(**dict(vars(overlap),parallel_observation=guard))
    scope=dict(vars(base),overlap=private_overlap,ROUND=ROUND,MODE=MODE,SCOPE=SCOPE)
    for name in ('source_gate','parallel_observation','accepted','append_registry'):
        scope[name]=overlap._clone_plain(getattr(base,name),scope)
    namespace=overlap._clone_plain(base._namespace,scope)(args)
    previous_process=namespace['run_process'];previous_durable=namespace['durable_json']
    previous_snapshot=namespace['_snapshot']
    def process(command,**options):
        if '--worker'in command:
            if command[:4]!=[str(host_python()),'-B','-m',original.MODULE]or 'pt_agent.tools.nixrun'in command:
                raise ValueError('Run only the unchanged original worker directly in the existing host runtime')
            options['env']=host_environment(options.get('env'))
        return previous_process(command,**options)
    def durable(path,result):
        if Path(path).name=='result.json'and result.get('round')==ROUND:
            result['host_runtime']=runtime_record(args)
            result['runtime_variant']='host-runtime-cpu-overlap'
            result['timing_variant']='overlap-with-identified-BERT-plus-existing-host-runtime'
            result['limits'].update(nix_seconds=None,legacy_Nix_seconds_not_applied=original.NIX_SECONDS)
        return previous_durable(path,result)
    def snapshot(out):
        rows=previous_snapshot(out);target=out/'source'/'heir-vgg-host-cpu-overlap.py'
        shutil.copyfile(__file__,target)
        rows.append(dict(original=str(Path(__file__).resolve()),snapshot=str(target),bytes=target.stat().st_size))
        return rows
    namespace.update(worker_command=worker_command,run_process=process,durable_json=durable,_snapshot=snapshot)
    return namespace


def execute(args):
    if Path(args.python).expanduser().absolute()!=host_python()or not host_python().is_file()or not os.access(host_python(),os.X_OK):
        raise ValueError('Use the already imported existing R16a Orion host Python; no new runtime is selected')
    # Keep every original parallel identity/path/resource validation. Only the
    # explicitly new result-prefix literal follows this runtime variant.
    tree=ast.parse(textwrap.dedent(inspect.getsource(base.execute)))
    old_prefix='heir-vgg16-fused-pool-conv-cpu-overlap-'
    if sum(isinstance(n,ast.Constant)and n.value==old_prefix for n in ast.walk(tree))!=1:
        raise ValueError('Original overlap result-prefix validation seam changed')
    namespace=dict(vars(base),_namespace=_namespace)
    cloned=overlap._clone_plain(base.execute,namespace)
    cloned=type(cloned)(cloned.__code__.replace(co_consts=tuple(OUT_PREFIX if v==old_prefix else v for v in cloned.__code__.co_consts)),
        namespace,cloned.__name__,cloned.__defaults__,cloned.__closure__)
    return cloned(args)


def parser():
    command=base.parser();command.description=__doc__;command.set_defaults(python=host_python());return command


def main(argv=None):
    import json
    result=execute(parser().parse_args(argv))
    print(json.dumps({k:result.get(k)for k in ('status','error','strict_math_passed','model_compiled','model_compiled_accepted','runtime_variant')}))
    return int(result['status']not in ('ok','prepared'))


if __name__=='__main__':raise SystemExit(main())
