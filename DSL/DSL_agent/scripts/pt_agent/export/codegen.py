"""One structured kernel call per lowered PT-DSL operation."""

from collections import defaultdict
import keyword

from ..lower import HANDLED_OPS, LoweringError


def generate(program) -> str:
    if not program.name.isidentifier() or keyword.iskeyword(program.name):
        raise ValueError("PT-DSL function name must be a Python identifier")
    consumers = defaultdict(list)
    for op in program.ops:
        for name in op.inputs:
            consumers[name].append(op)
    lines = [f'"""{program.name}: generated from program.ptdsl by pt_agent.export. Do not edit."""',
             "from . import ops", "", "", "def forward(b, W, A, "
             + ", ".join(param.name for param in program.params) + "):"]
    skipped = set()
    for op in program.ops:
        if op.id in skipped:
            continue
        a, output = op.attrs, op.outputs[0]
        args = ["b", *op.inputs]
        kind, comment = op.kind, ""
        if kind not in HANDLED_OPS:
            raise LoweringError(op.id, op.lineno, f"v1 lowering 不支持 {kind}")
        if kind == "pack":
            args += [f"layout={a['layout']!r}", f"gap={a['gap']!r}"]
        elif kind == "conv2d":
            args += ["W"]
            for key in ("w", "b", "bn", "bn_eps", "stride", "pad", "dilation", "groups", "algo"):
                value = a[key]
                value = value.name if key in ("w", "b") and value is not None else value
                args.append(f"{key}={value!r}")
        elif kind == "act":
            args += [f"A[{output!r}]"]
        elif kind == "slice":
            uses = consumers[output]
            if (a["axis"] != 2 or a["start"] != 0 or a["step"] != 2 or len(uses) != 1):
                raise LoweringError(op.id, op.lineno, "slice requires the downsample slice/slice/pad chain")
            second = uses[0]
            following = consumers[second.outputs[0]]
            if (second.kind != "slice" or second.attrs["axis"] != 3 or second.attrs["start"] != 0
                    or second.attrs["step"] != 2 or len(following) != 1 or following[0].kind != "pad"):
                raise LoweringError(op.id, op.lineno, "slice requires the downsample slice/slice/pad chain")
            pad = following[0]
            pads = pad.attrs["pads"]
            if len(pads) != 6 or pads[:4] != (0, 0, 0, 0):
                raise LoweringError(op.id, op.lineno, "downsample pad requires channel-only padding")
            kind, output = "downsample", pad.outputs[0]
            args += [f"pads={pads[4:]!r}"]
            comment = f"   # slice {op.outputs[0]}, slice {second.outputs[0]}, pad {output}"
            skipped.update((second.id, pad.id))
        elif kind == "pad":
            raise LoweringError(op.id, op.lineno, "standalone pad is unsupported")
        elif kind == "mean":
            args += [f"axis={a['axis']!r}", f"keepdim={a['keepdim']!r}"]
        elif kind == "relayout":
            args += [f"layout={a['layout']!r}"]
        elif kind == "linear":
            args += ["W", f"w={a['w'].name!r}", f"b={a['b'].name if a['b'] else None!r}",
                     f"algo={a['algo']!r}", f"n1={a['n1']!r}"]
        elif kind == "refresh":
            args += [f"enabled={program.header.refresh == 'agent'!r}"]
        lines.append(f"    {output} = ops.{kind}({', '.join(args)}){comment}")
    lines.append("    return ["+", ".join(program.outputs)+"]")
    return "\n".join(lines)+"\n"
