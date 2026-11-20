"""A new finite serial BERT comparison with three hours for each system.

The existing two-hour controller must close first.  One existing resource job
archives its redundant restored input, then a fresh original Agent pipeline and
an independent Rule63 native search run.  Their results remain independent.
"""
import argparse
import copy
from pathlib import Path
from types import FunctionType

from .. import paths
from . import autonomous_serial_batch as original
from . import bert_fresh_cold_3h as agent
from . import bert_rule63_shared_optimized_compile_3h as native
from . import bert_rule63_restored_input_archive as archive


MODULE = 'pt_agent.experiments.bert_three_hour_comparison_queue'


def option(argv, flag):
    if argv.count(flag) != 1 or argv.index(flag)+1 >= len(argv):
        raise ValueError('One explicit comparison option required: '+flag)
    return argv[argv.index(flag)+1]


def fixed_comparison(manifest, root, *, base):
    value = base(manifest, root)
    jobs = value['jobs']
    if (len(jobs) != 3 or value.get('direct_human_budget_seconds') != 10800
            or value.get('original_two_hour_batch_untouched') is not True):
        raise ValueError('Keep the explicitly selected three-job, three-hour comparison')
    required = [(archive.MODULE,1860),(agent.MODULE,10920),(native.MODULE,13320)]
    for job,(module,seconds) in zip(jobs,required):
        if job['argv'][3] != module or job['seconds'] != seconds:
            raise ValueError('Keep each original finite resource and scientific allocation')
    first = jobs[0]['id']
    if jobs[0].get('requires',[]) != [] or any(job.get('requires') != [first] for job in jobs[1:]):
        raise ValueError('Both scientific runs require resource preparation and remain independent')
    if (option(jobs[0]['argv'],'--seconds') != '1740'
            or jobs[0]['argv'].count('--no-future-restored-input-consumers') != 1
            or jobs[0]['argv'].count('--run') != 1
            or option(jobs[2]['argv'],'--native-seconds') != '10800'
            or option(jobs[2]['argv'],'--seconds') != '13200'):
        raise ValueError('The native search must receive its complete10800s allocation')
    for job in jobs[1:]:
        if option(job['argv'],'--paper-runs') != option(jobs[1]['argv'],'--paper-runs'):
            raise ValueError('Publish independent outcomes to the same existing paper ledger')
    watch = value['watch']
    if watch['pid'] <= 0 or watch['starttime'] <= 0:
        raise ValueError('Wait on the exact positive original two-hour controller birth')
    return value


def namespace():
    source = vars(original)
    values = dict(source, MODULES=set(original.MODULES)|
                  {agent.MODULE.removeprefix(original.PREFIX),native.MODULE.removeprefix(original.PREFIX)})
    for name,value in source.items():
        if isinstance(value,FunctionType) and value.__globals__ is source:
            clone=FunctionType(value.__code__,values,value.__name__,value.__defaults__,value.__closure__)
            clone.__kwdefaults__=copy.copy(value.__kwdefaults__)
            values[name]=clone
    validate=values['validate']
    values['validate']=lambda manifest,root:fixed_comparison(manifest,root,base=validate)
    return values


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--manifest',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--lease-wait-seconds',type=int,default=0)
    args=cli.parse_args(argv)
    args.manifest=args.manifest.expanduser().resolve()
    args.out=args.out.expanduser().resolve()
    return namespace()['run'](args)


if __name__=='__main__':
    raise SystemExit(main())
