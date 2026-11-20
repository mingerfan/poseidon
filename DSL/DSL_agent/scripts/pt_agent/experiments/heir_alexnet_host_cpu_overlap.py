"""Original Alex wide-pool mathematics/native compile in the existing host runtime.

Use the reviewed exact tune4 overlap supervision and unchanged original Alex
worker/source gates. Each real phase is one bounded attempt; root alone launches
after the VGG job closes and all live identities and resource checks still pass.
"""
from pathlib import Path
import shutil
from types import SimpleNamespace

from . import heir_alexnet_wide_pool as alex
from . import heir_vgg_host_cpu_overlap as host

ROUND='HEIR-ALEX-HOST-CPU-OVERLAP'
MODE='original-wide-pool-alex-training-math-native-existing-host-runtime-during-bert-tune4'
MODULE='pt_agent.experiments.heir_alexnet_host_cpu_overlap'
OUT_PREFIX='heir-alexnet-wide-pool-mask-host-cpu-overlap-'
SCOPE=('Original Alex wide-pool/interior-mask representation,31 emitted constants, '
    'three saved original training probes with21 activation pairs and30 logits under '
    'the original binary64 criterion, then original full native HEIR lowering. '
    'Exact identified BERT tune4/calls4 and original batch owner; CPU-overlap plus '
    'existing R16a Orion host-runtime variant. Original trained weights, program, '
    'learned biases, reference, native parameters and math policy retained. '
    'No Go, context, keys, CKKS, report, confirmation or API.')


def _scope():
    # The original Alex entry already supplies its own ordinary supervision
    # namespace. Keep all its worker/source/math/native callables unchanged.
    numeric=SimpleNamespace(**alex._namespace())
    base=SimpleNamespace(**dict(vars(host.base),original=numeric))
    scope=dict(vars(host),original=numeric,base=base,ROUND=ROUND,MODE=MODE,
        SCOPE=SCOPE,OUT_PREFIX=OUT_PREFIX)
    for name in ('worker_command','runtime_record'):
        scope[name]=host.overlap._clone_plain(getattr(host,name),scope)
    scope['_namespace']=lambda args:_supervised_namespace(args,scope)
    return scope


def _supervised_namespace(args,scope):
    namespace=host.overlap._clone_plain(host._namespace,scope)(args)
    previous_snapshot=namespace['_snapshot'];previous_durable=namespace['durable_json']
    def snapshot(out):
        rows=previous_snapshot(out);target=out/'source'/'heir-alexnet-host-cpu-overlap.py'
        shutil.copyfile(__file__,target)
        rows.append(dict(original=str(Path(__file__).resolve()),snapshot=str(target),bytes=target.stat().st_size))
        return rows
    def durable(path,result):
        if Path(path).name=='result.json'and result.get('round')==ROUND:
            result['host_runtime_variant_scope']='Original Alex wide-pool worker in previously observed existing R16a host runtime; actual Alex numerical/native outcome retained separately'
            result['original_worker_module']=alex.MODULE
        return previous_durable(path,result)
    namespace.update(_snapshot=snapshot,durable_json=durable)
    return namespace


def _namespace(args):return _supervised_namespace(args,_scope())


def execute(args):
    return host.overlap._clone_plain(host.execute,_scope())(args)


def parser():
    command=host.parser();command.description=__doc__;return command


def main(argv=None):
    import json
    result=execute(parser().parse_args(argv))
    print(json.dumps({key:result.get(key)for key in ('status','error','strict_math_passed',
        'model_compiled','model_compiled_accepted','original_worker_module','runtime_variant')}))
    return int(result['status']not in('ok','prepared'))


if __name__=='__main__':raise SystemExit(main())
