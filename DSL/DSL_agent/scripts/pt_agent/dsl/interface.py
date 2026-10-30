"""The model's original forward signature, shared by all r8-v8 entry points."""

INPUT_KINDS = {
    'tokens': dict(dsl_type='Cipher', pack_layouts=('token',)),
    'mask': dict(dsl_type='Mask', pack_layouts=('flat',)),
    'image': dict(dsl_type='Cipher', pack_layouts=('mux',)),
    'vector': dict(dsl_type='Cipher', pack_layouts=('flat',)),
}
SIGNATURE_RULE = ('Use every item in inputs as a function parameter with exactly its name, '
                  'order, DSL type and shape. Preserve the complete signature; pack each '
                  'input using one of its pack_layouts.')


def inputs(info):
    result = []
    for item in info.inputs:
        shape = tuple(item['shape'])
        # Old synthetic ModelInfo fixtures predate the inferred kind field.
        kind = item.get('kind') or {2: 'vector', 3: 'tokens', 4: 'image'}.get(len(shape))
        spec = INPUT_KINDS[kind]
        result.append(dict(name=item['name'], shape=list(shape), kind=kind,
                           dsl_type=spec['dsl_type'], pack_layouts=list(spec['pack_layouts'])))
    return result


def check_interface(program, info):
    expected = inputs(info)
    actual = [dict(name=p.name, shape=list(p.shape),
                   dsl_type={'cipher': 'Cipher', 'mask': 'Mask', 'plain': 'Plain'}[p.kind])
              for p in program.params]
    errors = []
    for index in range(max(len(expected), len(actual))):
        wanted = expected[index] if index < len(expected) else None
        got = actual[index] if index < len(actual) else None
        if wanted is None or got is None or any(got[key] != wanted[key] for key in ('name', 'shape', 'dsl_type')):
            errors.append(dict(code='E_INTERFACE', parameter=index, expected=wanted, actual=got,
                               message=SIGNATURE_RULE))
    return dict(ok=not errors, expected=expected, actual=actual, errors=errors)


def require_interface(program, info):
    result = check_interface(program, info)
    if not result['ok']:
        import json
        raise ValueError('Input interface check failed: '+json.dumps(result))
    return result
