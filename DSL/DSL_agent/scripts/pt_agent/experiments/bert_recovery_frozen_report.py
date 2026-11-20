"""Original isolated872 report bound to the recovered cold three-hour owner."""
import copy
import json
from pathlib import Path
from types import FunctionType

from . import bert_fresh_frozen_report_3h as original
from . import bert_supervisor_recovery as recovery

ROUND = 'BERT-FRESH-3H-RECOVERY-FROZEN-REPORT'
MODEL = original.MODEL
MODE = 'isolated-recovered-fresh-three-hour-accepted-native-validation872-once'
MODULE = 'pt_agent.experiments.bert_recovery_frozen_report'
base, paths, fresh = original.base, original.paths, original.fresh
_option = original._option


def queue_registry_row(owner, run):
    row = original.registry_rows(owner.parent/'registry.jsonl', {owner.name:owner})[owner.name]
    recovery.require(row.get('round') == run.get('round') and row.get('mode') == run.get('mode')
        and row.get('model') == 'original-model-experiment-queue'
        and row.get('metrics') == dict(status=run.get('status'), jobs=run.get('jobs')),
        'Unchanged actual recovered owner registry and terminal jobs required')
    recovery.namespace(limits={})['validate'](base.read(owner/'manifest.json'), paths.RESULTS_DIR)
    return row


def admission_namespace():
    source = vars(original); values = dict(source, queue_registry_row=queue_registry_row)
    for name, fn in source.items():
        if isinstance(fn, FunctionType) and fn.__globals__ is source:
            clone=FunctionType(fn.__code__,values,fn.__name__,fn.__defaults__,fn.__closure__)
            clone.__kwdefaults__=copy.copy(fn.__kwdefaults__); values[name]=clone
    values['queue_registry_row'] = queue_registry_row
    return values


def admission(args):
    recovery.require(args.source_case.name == recovery.FRESH.replace('-20261117','-logquota-recovery-20261117')
        and args.supervisor_case.name == 'bert-native10800-then-fresh-agent-logquota-recovery-serial-20261117',
        'Bind only the new independent recovered cold source and its actual owner')
    return admission_namespace()['admission'](args)


def namespace():
    values=original.namespace()
    values.update(ROUND=ROUND,MODE=MODE,MODULE=MODULE,admission=admission)
    return values


def main(argv=None):
    args=original.parser().parse_args(argv)
    for name,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,name,value.expanduser().absolute() if name.endswith('python') else value.expanduser().resolve())
    args.compile_case=args.source_case
    result=namespace()['execute'](args)
    print(json.dumps({key:result.get(key) for key in ('status','report_accepted','report_samples','error')}),flush=True)
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
