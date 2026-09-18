"""Python-style argument binding for symbolic construction functions only."""
from seal_artifact_gate import require


def parameter_nodes(args):
    return [*args.posonlyargs, *args.args, *([args.vararg] if args.vararg else []),
            *args.kwonlyargs, *([args.kwarg] if args.kwarg else [])]


def bind(args, defaults, keyword_defaults, positional, keywords):
    """Values are already evaluated; never copy mutable default/argument values."""
    positional_names = [p.arg for p in [*args.posonlyargs, *args.args]]
    posonly = {p.arg for p in args.posonlyargs}
    kwonly = {p.arg for p in args.kwonlyargs}
    require(len(positional) <= 128 and len(keywords) <= 128, 'Call argument resource limit')
    require(args.vararg is not None or len(positional) <= len(positional_names), 'Too many positional arguments')
    result = dict(zip(positional_names, positional))
    if args.vararg:
        result[args.vararg.arg] = tuple(positional[len(positional_names):])
    extra = {}
    for name, value in keywords.items():
        if name in posonly:
            require(args.kwarg is not None, 'Positional-only argument passed by keyword: '+name)
            extra[name] = value
        elif name in positional_names or name in kwonly:
            require(name not in result, 'Multiple values for argument: '+name)
            result[name] = value
        else:
            require(args.kwarg is not None, 'Unexpected keyword argument: '+name)
            extra[name] = value
    start = len(positional_names) - len(defaults)
    for i, name in enumerate(positional_names):
        if name not in result:
            require(i >= start, 'Missing required positional argument: '+name)
            result[name] = defaults[i-start]
    kw_defaults = dict(keyword_defaults)
    for parameter in args.kwonlyargs:
        name = parameter.arg
        if name not in result:
            require(name in kw_defaults, 'Missing required keyword-only argument: '+name)
            result[name] = kw_defaults[name]
    if args.kwarg:
        result[args.kwarg.arg] = extra
    return result
