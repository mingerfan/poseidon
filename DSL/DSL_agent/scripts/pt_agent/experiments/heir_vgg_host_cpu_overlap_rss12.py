"""Finite RSS12 continuation of the original host VGG export/math/native case.

Preserve the original worker, constant parser, strict mathematics, source and
native parameters. Only own aggregate physical RSS allowance changes from10 to12.
"""
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from . import heir_vgg_host_cpu_overlap as host

ROUND='HEIR-VGG-HOST-CPU-OVERLAP-RSS12'
MODE=host.MODE+'-rss12'
MODULE='pt_agent.experiments.heir_vgg_host_cpu_overlap_rss12'
OUT_PREFIX='heir-vgg16-fused-pool-conv-host-cpu-overlap-rss12-'
RSS_GIB=12
GIB=host.GIB
RSS4_REASONS=host.base.RSS4_REASONS
overlap=host.overlap
SCOPE=(host.SCOPE+' New independent attempt retains AS24GiB and hostavailable2GiB; '
    'only own aggregate RSS allowance is12GiB after the recorded RSS10 guard stop.')


def parallel_observation(args,sample,available,free):
    """Original shared predicates and RSS4 replacement, now exactly RSS12."""
    row=overlap.parallel_observation(args,sample,available,free)
    original_reasons=list(row['reasons'])
    row['shared_guard_scope']=row['scope']
    row['shared_guard_reasons_before_RSS_policy']=original_reasons
    row['reasons']=[reason for reason in original_reasons if reason not in RSS4_REASONS]
    rss=row['subtree_rss_bytes']
    if rss is not None and rss>RSS_GIB*GIB:
        row['reasons'].append('Own supervisor and descendant RSS exceeds12GiB')
    row.update(stop=bool(row['reasons']),scope=SCOPE,own_aggregate_RSS_gib=RSS_GIB,
        observed_perf_counter_ns=time.perf_counter_ns())
    return row


def _scope():
    base=SimpleNamespace(**dict(vars(host.base),RSS_GIB=RSS_GIB,parallel_observation=parallel_observation))
    scope=dict(vars(host),base=base,ROUND=ROUND,MODE=MODE,SCOPE=SCOPE,OUT_PREFIX=OUT_PREFIX)
    scope['_namespace']=lambda args:_supervised_namespace(args,scope)
    return scope


def _supervised_namespace(args,scope):
    namespace=host.overlap._clone_plain(host._namespace,scope)(args)
    previous_snapshot=namespace['_snapshot'];previous_durable=namespace['durable_json']
    def snapshot(out):
        rows=previous_snapshot(out);target=out/'source'/'heir-vgg-host-cpu-overlap-rss12.py'
        shutil.copyfile(__file__,target)
        rows.append(dict(original=str(Path(__file__).resolve()),snapshot=str(target),bytes=target.stat().st_size))
        return rows
    def durable(path,result):
        if Path(path).name=='result.json'and result.get('round')==ROUND:
            result['resource_variant']='aggregate-RSS12GiB; original-AS24/hostavailable2GiB'
            result['prior_RSS10_failure_preserved']=True
        return previous_durable(path,result)
    namespace.update(_snapshot=snapshot,durable_json=durable)
    return namespace


def _namespace(args):return _supervised_namespace(args,_scope())


def execute(args):return host.overlap._clone_plain(host.execute,_scope())(args)


def parser():
    command=host.parser();command.description=__doc__;return command


def main(argv=None):
    import json
    result=execute(parser().parse_args(argv))
    print(json.dumps({k:result.get(k)for k in ('status','error','strict_math_passed',
        'model_compiled','model_compiled_accepted','resource_variant')}))
    return int(result['status']not in('ok','prepared'))


if __name__=='__main__':raise SystemExit(main())
