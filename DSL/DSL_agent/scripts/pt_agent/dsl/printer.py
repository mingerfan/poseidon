"""Canonical expanded PT-DSL text, with deterministic value names."""

import json

from .ir import Program, WeightRef


def _literal(value):
    if isinstance(value, WeightRef):
        return "W[" + json.dumps(value.name) + "]"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, tuple):
        return "(" + ", ".join(_literal(x) for x in value) + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def print_program(program: Program) -> str:
    header = program.header
    settings = ", ".join(f"{name}={_literal(getattr(header, name))}"
                         for name in ("slots", "refresh", "placement", "l_max", "boot_out_level")
                         if getattr(header, name) is not None)
    if header.manage != "compiler":
        settings += ", manage="+_literal(header.manage)
    params = ", ".join(p.name + ": " + ("Cipher" if p.kind == "cipher" else "Plain")
                       + "[" + ", ".join(map(str, p.shape)) + "]" for p in program.params)
    lines = [f"@fhe.program({settings})", f"def {program.name}({params}):"]
    names = {p.name: p.name for p in program.params}
    index, active_region = 0, None
    for op in program.ops:
        if op.region != active_region and op.region is not None:
            region = program.regions[op.region]
            lines.append(f"    with region({_literal(region.name)}, level_budget={region.level_budget}):")
        active_region = op.region
        args = [names[x] for x in op.inputs]
        outputs = []
        for vid in op.outputs:
            names[vid] = f"v{index}"
            outputs.append(names[vid])
            index += 1
        args += [f"{key}={_literal(value)}" for key, value in op.attrs.items() if value is not None]
        indent = "        " if op.region is not None else "    "
        lines.append(indent + ", ".join(outputs) + f" = {op.kind}(" + ", ".join(args) + ")")
    lines.append("    return " + ", ".join(names[x] for x in program.outputs))
    return "\n".join(lines) + "\n"
