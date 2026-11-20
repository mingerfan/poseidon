"""Original Alex JSON worker/native pipeline under its ancestor's serial lease.

Reuse the original Alex ordinary supervisor and initializer contextmanager. The
global heavy lease must belong to one fixed live ancestor; no BERT phase, worker
identity or session is consulted. Root schedules one fresh bounded case.
"""
from pathlib import Path
import json
import os
import shutil
import time
from types import SimpleNamespace

from .. import paths
from . import heir_alexnet_host_cpu_overlap_json as numeric
from . import heir_vgg_cpu_overlap as registry_helper

alex=numeric.alex
host=numeric.host
shared=host.overlap
original=host.original
ROUND='HEIR-ALEX-HOST-SERIAL-JSON'
MODE='original-Alex-wide-pool-strict-math-native-host-runtime-serial-JSON'
MODULE='pt_agent.experiments.heir_alexnet_host_serial_json'
OUT_PREFIX='heir-alexnet-wide-pool-mask-host-serial-json-'
SCOPE=('Original Alex31-constant/three-training-input strict mathematics and '
    'complete native compilation, using the already reviewed private numeric JSON '
    'worker and existing R16a host runtime. Serial clock under one fixed original '
    'ancestor holding the unique global heavy lease; no BERT/session dependency. '
    'Original weights/program/reference/native parameters and mathematical '
    'criterion retained; no Go, context, keys, CKKS, confirmation, report or API.')


def lease_observation(expected=None):
    table=shared._scan();own=os.getpid();row=table.get(own)or{}
    if row.get('pid')!=own or row.get('state')not in('R','S','D'):
        raise ValueError('Own live process identity must be observed in the host namespace')
    ancestors=[];seen={own};pid=row.get('ppid')
    while pid in table and pid not in seen:
        seen.add(pid);ancestors.append(pid);pid=table[pid].get('ppid')
    owners=shared._lease_owners(paths.RESULTS_DIR/'autonomous-heavy-experiments.lock')
    if len(owners)!=1 or owners[0]not in ancestors:
        raise ValueError('The unique global heavy lease owner must be an original strict ancestor')
    owner=table[owners[0]]
    if owner.get('state')not in('R','S','D')or type(owner.get('starttime'))is not int or owner['starttime']<=0:
        raise ValueError('Original live ancestor lease-owner birth identity required')
    identity=dict(pid=owners[0],starttime=owner['starttime'])
    if expected is not None and identity!=expected:raise ValueError('The original ancestor lease owner changed')
    family=shared._family(table,own)
    foreign=[dict(pid=p,starttime=r.get('starttime'),name=shared._process_name(p))for p,r in table.items()
        if p not in family and r.get('state')not in('Z','X')and shared._process_name(p)in shared.COMPILERS]
    if foreign:raise ValueError('Another native compiler is active outside this serial case')
    return dict(owner=identity,ancestors=ancestors,own_pid=own,
        own_subtree_rss_bytes=sum(table[p].get('vmrss_kib',0)*1024 for p in family),
        supervisor_rss_bytes=row.get('vmrss_kib',0)*1024,foreign_compilers=foreign)


def worker_command(source,out,interpreter,memory_gib=24):
    if alex.validate_memory_gib(memory_gib)!=24 or Path(interpreter).absolute()!=host.host_python():
        raise ValueError('Original AS24 and existing R16a host interpreter required')
    command=[str(interpreter),'-B','-m',numeric.MODULE,'--worker','--out',str(out),'--memory-gib','24']
    for name in original.SOURCE_FIELDS:command+=['--'+name.replace('_','-'),str(source[name])]
    return command


def _namespace(args):
    namespace=alex._namespace();identity=[None];latest=[None];deadline=time.monotonic()+alex.TOTAL_SECONDS
    def observe(sample,available,free):
        row=original.physical_observation(sample,available,free)
        try:
            lease=lease_observation(identity[0]);identity[0]=lease['owner']
            rss=max(lease['own_subtree_rss_bytes'],(row['subtree_rss_bytes']or 0)+lease['supervisor_rss_bytes'])
            row.update(serial_lease=lease,subtree_rss_bytes=rss)
            if rss>14*original.GIB:row['reasons'].append('Own supervisor and descendant RSS exceeds14GiB')
        except(OSError,ValueError,KeyError,TypeError)as exc:row['reasons'].append(type(exc).__name__+': '+str(exc))
        row.update(stop=bool(row['reasons']),scope=SCOPE);latest[0]=row;return row
    def source_gate(source):
        checked=alex.source_gate(source);registry=args.registry or paths.RESULTS_DIR/'registry.jsonl'
        if registry.exists():
            for line in registry.read_text().splitlines():
                if not line.strip():continue
                row=json.loads(line)
                if(row.get('round')==ROUND and row.get('mode')==MODE
                        and any(((row.get('metrics')or{}).get('attempted')or{}).get(k,0)>0 for k in('export_math','native_compile'))):
                    raise ValueError('This explicit serial host variant already attempted once')
        return checked
    def durable(path,result):
        if Path(path).name=='result.json'and result.get('round')==ROUND:
            if result.get('status')not in('running','prepared','ok','error'):
                result['original_termination_status']=result['status'];result['status']='error'
            result.update(scope=SCOPE,runtime_variant='existing-host-runtime-serial',timing_variant='fresh-serial-host-runtime',
                constant_parser_variant='private-Alex-mask-and-common-numeric-JSON',
                host_runtime=host.runtime_record(args),ancestor_heavy_lease_owner=identity[0],
                latest_serial_observation=latest[0],BERT_session_dependency=False,
                existing_initializer_lock_retained=True,model_compiled_accepted=registry_helper.accepted(result))
            result['limits'].update(nix_seconds=None,legacy_Nix_seconds_not_applied=original.NIX_SECONDS)
        original.durable_json(path,result)
    def process(command,**options):
        if options['mem_gib']!=24 or options['fsize_gib']!=16:raise ValueError('Preserve original AS24/file16')
        if '--worker'in command:
            if command[:4]!=[str(host.host_python()),'-B','-m',numeric.MODULE]:raise ValueError('Original reviewed JSON worker required')
            options['env']=host.host_environment(options.get('env'))
        result=original.run_process(command,**options)
        original.durable_json(Path(options['log']).parent/'process-return.json',result);return result
    def snapshot(out):
        rows=alex._snapshot(out)
        for path,label in((Path(__file__),'heir-alexnet-host-serial-json.py'),
                (Path(numeric.__file__),'original-private-Alex-JSON-worker.py'),
                (Path(numeric.common_json.__file__),'original-private-common-JSON-checker.py'),
                (Path(shared.__file__),'shared-public-lease-metadata.py')):
            target=out/'source'/label;shutil.copyfile(path,target)
            rows.append(dict(original=str(path.resolve()),snapshot=str(target),bytes=target.stat().st_size))
        return rows
    namespace.update(ROUND=ROUND,MODE=MODE,SCOPE=SCOPE,source_gate=source_gate,
        worker_command=worker_command,physical_observation=observe,durable_json=durable,
        run_process=process,_snapshot=snapshot,accepted=registry_helper.accepted)
    namespace['execute']=shared._clone_plain(namespace['execute'],namespace)
    old_terminal=shared._clone_plain(namespace['terminal_result'],namespace)
    def terminal(out,result):
        try:
            row=observe(None,original._available_ram_bytes(),shutil.disk_usage(out).free)
            if row['stop']:result.update(status='error',error='; '.join(row['reasons']),terminal_serial_guard=row)
            if time.monotonic()>=deadline:result.update(status='error',error='Original10800s serial case allowance exhausted at closure',terminal_deadline_exhausted=True)
        except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
        old_terminal(out,result)
    namespace['terminal_result']=terminal
    namespace['original']=SimpleNamespace(**namespace)
    namespace['append_registry']=shared._clone_plain(registry_helper.append_registry,namespace)
    return namespace


def execute(args):
    args.out=args.out.expanduser().resolve();args.python=args.python.expanduser().absolute()
    if(args.out.parent!=paths.RESULTS_DIR.resolve()or not args.out.name.startswith(OUT_PREFIX)
            or args.memory_gib!=24 or args.python!=host.host_python()
            or not args.python.is_file()or not os.access(args.python,os.X_OK)or getattr(args,'worker',False)):
        raise ValueError('A fresh own serial case, AS24 and existing host interpreter are required')
    return _namespace(args)['execute'](args)


def parser():
    command=alex.parser();command.description=__doc__
    command.set_defaults(python=host.host_python(),heir_opt=paths.WORKSPACE/'baselines/heir-2026.11.1-venv/bin/heir-opt')
    return command


def main(argv=None):
    import json
    result=execute(parser().parse_args(argv));print(json.dumps({k:result.get(k)for k in('status','error',
        'strict_math_passed','model_compiled','model_compiled_accepted','runtime_variant')}))
    return int(result['status']not in('ok','prepared'))


if __name__=='__main__':raise SystemExit(main())
