"""Fixed public descriptor namespaces, with no pointer traversal or DSL evaluation."""
from benchmark_graph import require
from public_origin_columns import descriptor_references

POLICY = 'descriptors-namespace-refs-v1'
FORMAT = 'poseidon-public-weight-namespace-descriptors-v1'
NAMES = ('model_constants', 'public_constants')
RULES = """\nPublic descriptor namespace references v1:
Model weight descriptors occur once in model.constants. Prebound public Plain
value descriptors occur once in public_constants. The two fields with those names
inside public_weight_binding are fixed namespace references with content_sha256,
referring exactly to model.constants and public_constants. Hashes bind the complete
respective descriptor registry. There are no other namespaces, chained references,
file reads or replacement coefficients. All shapes, names, content hashes, original
constant_origins and computation constraints remain present and unchanged.
\n"""


def is_projected(context):
    if type(context) is not dict or type(context.get('public_weight_binding')) is not dict:
        return False
    info = context['public_weight_binding']
    return ((info.get('format'), info.get('policy')) == (FORMAT, POLICY) and
            any(type(info.get(name)) is dict and type(info[name].get('source')) is str
                for name in NAMES))


def references(model_constants, public_constants):
    # Reuse the existing exact registry digest contract; do not invent aliases.
    return descriptor_references(model_constants, public_constants)


def registries(context):
    require(type(context) is dict and type(context.get('public_weight_binding')) is dict and
            type(context.get('model')) is dict and type(context['model'].get('constants')) is dict and
            type(context.get('public_constants')) is dict, 'Public descriptor namespace fields')
    info = context['public_weight_binding']
    require((info.get('format'), info.get('policy')) == (FORMAT, POLICY),
            'Public descriptor namespace policy')
    expected = references(context['model']['constants'], context['public_constants'])
    require(all(type(info.get(name)) is dict and set(info[name]) == {'source', 'content_sha256'} and
                info[name] == expected[name] for name in NAMES),
            'Public descriptor namespace reference/hash')
    return dict(model_constants=context['model']['constants'], public_constants=context['public_constants'])
