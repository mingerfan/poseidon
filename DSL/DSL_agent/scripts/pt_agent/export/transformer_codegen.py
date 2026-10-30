"""One structured U4 operation per line in a standalone package."""

from ..dsl.ir import WeightRef,ValueRef


def record(value):
    if isinstance(value,WeightRef):return {'weight':value.name}
    if isinstance(value,ValueRef):return {'value':value.name}
    if isinstance(value,dict):return {k:record(v) for k,v in value.items()}
    return value


def generate(program):
    lines=[f'"""{program.name}: trusted token DSL lowering."""','from . import ops','','',
           'def forward(b,W,A,'+', '.join(p.name for p in program.params)+'):']
    for op in program.ops:
        if op.kind=='pack':
            expr=f'ops.pack(b,{op.inputs[0]},layout={op.attrs["layout"]!r},gap=1)'
        else:
            attrs=record(op.attrs)
            if op.kind=='matmul':
                following=[n for n in program.ops if op.outputs[0] in n.inputs]
                before=program.producer(op.inputs[0])
                if len(following)==1 and following[0].kind=='softmax':attrs['__keep__']=following[0].attrs['keep']
                elif before is not None and before.kind=='softmax':attrs['__keep__']=before.attrs['keep']
            inputs='{'+', '.join(repr(v)+':'+v for v in op.inputs)+'}'
            expr=f'ops.token(b,W,A,{inputs},input_names={op.inputs!r},kind={op.kind!r},attrs={attrs!r},output={op.outputs[0]!r})'
        lines.append('    '+op.outputs[0]+' = '+expr)
    lines.append('    return ['+', '.join(program.outputs)+']')
    return '\n'.join(lines)+'\n'
