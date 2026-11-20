"""Observe the recovered three-hour source and publish separate scalar files."""
import copy
from types import FunctionType

from . import bert_three_hour_metadata_observer as original
from . import bert_recovery_terminal_metadata as metadata

MODULE='pt_agent.experiments.bert_recovery_metadata_observer'
RENAMES={'bert_3h_comparison_terminal.json':'bert_3h_recovery_comparison_terminal.json',
         'bert_fresh_3h_final_quality.json':'bert_fresh_3h_recovery_final_quality.json'}


def namespace():
    source=vars(original);evidence=metadata.namespace();values=dict(source,evidence=evidence)
    for name,fn in source.items():
        if isinstance(fn,FunctionType)and fn.__globals__ is source:
            clone=FunctionType(fn.__code__,values,fn.__name__,fn.__defaults__,fn.__closure__)
            clone.__kwdefaults__=copy.copy(fn.__kwdefaults__);values[name]=clone
    def publish(manifest,name,data):
        name=RENAMES.get(name,name)
        for directory in manifest['destinations']:evidence.save(evidence.Path(directory)/name,data)
    values['publish']=publish
    return values


def main(argv=None):return namespace()['main'](argv)


if __name__=='__main__':raise SystemExit(main())
