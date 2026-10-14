"""Explicit public expression partitions; separate from native function witnesses."""
import ast
from seal_artifact_gate import require

# Expression, independently known public value. No model/reference rewriting.
RECIPES = {
    "binary.Add": ("0.25 + 0.5", .75),
    "binary.Sub": ("0.75 - 0.25", .5),
    "binary.Mult": ("0.25 * 2", .5),
    "binary.Div": ("1 / 2", .5),
    "binary.FloorDiv": ("5 // 2", 2),
    "binary.Mod": ("5 % 2", 1),
    "binary.Pow": ("0.5 ** 2", .25),
    "bool.And": ("0.25 and 0.5", .5),
    "bool.Or": ("0.0 or 0.5", .5),
    "unary.Not": ("float(not False)", 1),
    "compare.In": ("float(2 in [1,2])", 1),
    "compare.NotIn": ("float(3 not in [1,2])", 1),
    "call.int": ("int('2')", 2),
    "call.float": ("float('0.5')", .5),
    "call.len": ("len([1,2])", 2),
    "call.pow": ("pow(0.5,2)", .25),
    "call.floor": ("np.floor(1.75)", 1),
    "call.ceil": ("np.ceil(0.25)", 1),
    "call.log2": ("np.log2(2.0)", 1),
    "call.strip": ("float(' 0.5 '.strip())", .5),
    "call.lstrip": ("float(' 0.5'.lstrip())", .5),
    "call.rstrip": ("float('0.5 '.rstrip())", .5),
    "call.split": ("float('0.5,2'.split(',')[0])", .5),
    "call.rsplit": ("float('2,0.5'.rsplit(',')[1])", .5),
    "call.partition": ("float('0.5,2'.partition(',')[0])", .5),
    "call.rpartition": ("float('2,0.5'.rpartition(',')[2])", .5),
    "call.replace": ("float('0x5'.replace('x','.'))", .5),
    "call.join": ("float('.'.join(['0','5']))", .5),
    "call.array": ("np.array([0.5])[0]", .5),
    "call.asarray": ("np.asarray([0.5])[0]", .5),
    "call.reshape": ("np.array([[0.5,0.25]]).reshape(2)[0]", .5),
    "call.flatten": ("np.array([[0.5,0.25]]).flatten()[0]", .5),
    "call.copy": ("np.array([0.5],dtype=object).copy()[0]", .5),
    "call.transpose": ("np.array([[0.5,0.25]],dtype=object).transpose()[0,0]", .5),
    "attr.T": ("np.array([[0.5,0.25]],dtype=object).T[0,0]", .5),
    "attr.shape": ("np.array([[0.5,0.25]]).shape[1]", 2),
    "attr.size": ("np.array([0.5,0.25],dtype=object).size", 2),
    "attr.ndim": ("np.array([[0.5,0.25]],dtype=object).ndim", 2),
    "index.read": ("[0.5,0.25][0]", .5),
    "slice.read": ("[0.5,0.25][0:1][0]", .5),
}
# Typed partitions reuse the fixed scalar observer, not source spelling.
TYPED_RECIPES = {
    "cast.float.bool": ("float(True)",1),
    "cast.float.default": ("float()",0),
    "cast.float.numeric.legacy": ("float(np.array([0.5]))",.5),
    "cast.float.numeric.zero": ("float(np.array(0.5))",.5),
    "cast.float.object.legacy": ("float(np.array([0.5],dtype=object))",.5),
    "cast.float.object.zero": ("float(np.array(0.5,dtype=object))",.5),
    "cast.float.str": ("float('0.5')",.5),
    "cast.int.base": ("int('10',2)",2),
    "cast.int.bool": ("int(True)",1),
    "cast.int.default": ("int()",0),
    "cast.int.numeric.legacy": ("int(np.array([1.75]))",1),
    "cast.int.numeric.negative_fraction": ("int(np.array(-1.75))",-1),
    "item.numeric.flat_negative": ("np.array([[0.25,0.5]]).item(-1)",.5),
    "item.numeric.positional": ("np.array([[0.25,0.5]]).item(0,1)",.5),
    "item.numeric.sole": ("np.array([0.5]).item()",.5),
    "item.numeric.tuple": ("np.array([[0.25,0.5]]).item((0,1))",.5),
    "item.object.cipher": ("np.array([covered_input],dtype=object).item()",None),
    "object.named_numeric_constructor": ("float(np.array(SUPPLIED_HALF,dtype=object))",.5),
}
RECIPES.update(TYPED_RECIPES)
RECIPES.update({
    "call.dict": ("dict(v=0.5)['v']",.5),
    "call.enumerate": ("list(enumerate([0.5]))[0][1]",.5),
    "call.get": ("{'v':0.5}.get('v')",.5),
    "call.items": ("list({'v':0.5}.items())[0][1]",.5),
    "call.iter": ("next(iter([0.5,0.25]))",.5),
    "call.keys": ("float(list({'0.5':1}.keys())[0])",.5),
    "call.list": ("list((0.5,0.25))[0]",.5),
    "call.next": ("next(iter([0.5,0.25]))",.5),
    "call.pop": ("{'v':0.5}.pop('v')",.5),
    "call.popitem": ("{'v':0.5}.popitem()[1]",.5),
    "call.reversed": ("list(reversed([0.25,0.5]))[0]",.5),
    "call.setdefault": ("{}.setdefault('v',0.5)",.5),
    "call.sorted": ("sorted([0.75,0.5])[0]",.5),
    "call.tuple": ("tuple([0.5,0.25])[0]",.5),
    "call.values": ("list({'v':0.5}.values())[0]",.5),
    "call.zip": ("list(zip([0.25],[0.5]))[0][1]",.5),
    "call.Chebyshev": ("np.polynomial.Chebyshev([0.25,0.5]).coef[1]",.5),
    "attr.coef": ("np.polynomial.Chebyshev([0.25,0.5]).coef[1]",.5),
    "attr.domain": ("np.polynomial.Chebyshev([0.25,0.5],domain=[0.25,0.5]).domain[1]",.5),
    "attr.window": ("np.polynomial.Chebyshev([0.25,0.5],window=[0.25,0.5]).window[1]",.5),
})
TYPED_RULES = {
    "cast.float.bool":"Convert a public bool to float.",
    "cast.float.default":"Use zero-argument float().",
    "cast.float.numeric.legacy":"Convert a rank-positive singleton numeric array to float with pinned legacy behavior.",
    "cast.float.numeric.zero":"Convert a rank-zero numeric array to float.",
    "cast.float.object.legacy":"Convert a rank-positive singleton public object array to float.",
    "cast.float.object.zero":"Convert a rank-zero public object array to float.",
    "cast.float.str":"Convert a public string to float.",
    "cast.int.base":"Use int(public_string, explicit_public_base).",
    "cast.int.bool":"Convert a public bool to int.",
    "cast.int.default":"Use zero-argument int().",
    "cast.int.numeric.legacy":"Convert a rank-positive singleton numeric array to int.",
    "cast.int.numeric.negative_fraction":"Truncate a negative noninteger numeric-array value toward zero with int().",
    "item.numeric.flat_negative":"Use a negative flat item index on a multi-element numeric array.",
    "item.numeric.positional":"Use multiple positional item indices on a numeric array.",
    "item.numeric.sole":"Extract the sole numeric array cell with item() without arguments.",
    "item.numeric.tuple":"Use item((i,j)) on a rank-two numeric array.",
    "item.object.cipher":"Extract a whole ciphertext Expr from object storage with item(); no scalar conversion or decryption.",
    "object.named_numeric_constructor":"Convert an original supplied named public constant to object storage with its numeric shape.",
}
# Stateful templates use a known public bias; the model/reference never change.
STATE_RECIPES={
    "node.While":("coverage_bias=0\ni=0\nwhile i<2:\n    coverage_bias+=0.25\n    i+=1",.5),
    "node.Break":("coverage_bias=0\nfor i in range(3):\n    coverage_bias+=0.25\n    if i==1:\n        break",.5),
    "node.Continue":("coverage_bias=0\nfor i in range(3):\n    if i==1:\n        continue\n    coverage_bias+=0.25",.5),
    "event.for_else":("coverage_bias=0\nfor i in range(1):\n    coverage_bias+=0.25\nelse:\n    coverage_bias+=0.5",.75),
    "event.while_else":("coverage_bias=0\ni=0\nwhile i<2:\n    coverage_bias+=0.125\n    i+=1\nelse:\n    coverage_bias+=0.5",.75),
    "subscript.write":("a=[0.25]\na[0]=0.5\ncoverage_bias=a[0]",.5),
    "slice.write":("a=[0.25,0.125]\na[0:1]=[0.5]\ncoverage_bias=a[0]",.5),
    "aug.Add":("coverage_bias=0.25\ncoverage_bias+=0.25",.5),
    "aug.Mult":("coverage_bias=0.25\ncoverage_bias*=2",.5),
    "call.clear":("a={'v':0.5}\na.clear()\ncoverage_bias=float(len(a))",0),
    "call.update":("a={'v':0.25}\na.update({'v':0.5})\ncoverage_bias=a['v']",.5),
    "node.Pass":("coverage_bias=0.5\npass",.5),
    "node.Lambda":("f=lambda v:v+0.25\ncoverage_bias=f(0.25)",.5),
    "counter.default_evaluations":("def f(v=0.5):\n    return v\ncoverage_bias=f()",.5),
    "counter.closure_instances":("def f():\n    return 0.5\ncoverage_bias=f()",.5),
    "counter.helper_calls":("def f(v):\n    return v+0.25\ncoverage_bias=f(0.25)",.5),
    "counter.nonlocal_writes":("coverage_bias=0.25\ndef f():\n    nonlocal coverage_bias\n    coverage_bias=0.5\nf()",.5),
    "counter.starred_expansions":("def f(v):\n    return v\ncoverage_bias=f(*[0.5])",.5),
}
RECIPES.update({
    "node.ListComp":("[v+0.25 for v in [0.25]][0]",.5),
    "node.DictComp":("{v:v+0.25 for v in [0.25]}[0.25]",.5),
    "call.full":("np.full(1,covered_input,dtype=object)[0]",None),
    "call.concatenate":("np.concatenate((np.array([covered_input],dtype=object),np.array([covered_input],dtype=object)))[0]",None),
    "call.Empty":("(np.array([Empty()],dtype=object)+np.array([covered_input],dtype=object))[0]",None),
})
COUNTER_CONTEXTS={
    "comprehensions":"node.ListComp", "container_writes":"subscript.write",
    "iterator_advances":"call.next", "keyword_arguments":"call.dict",
    "lambda_instances":"node.Lambda", "loop_breaks":"node.Break",
    "loop_continues":"node.Continue", "loop_iterations":"node.While",
    "mapping_calls":"call.get", "mapping_views":"call.values", "mapping_writes":"call.update",
    "membership_tests":"compare.In", "object_array_operations":"call.copy",
    "public_array_operations":"call.array", "public_numeric_operations":"binary.Add",
    "public_numpy_functions":"call.floor", "public_polynomial_operations":"call.Chebyshev",
    "public_string_calls":"call.strip", "slice_writes":"slice.write",
    "while_iterations":"node.While", "empty_operations":"call.Empty",
}
for counter,context in COUNTER_CONTEXTS.items():
    key="counter."+counter
    if context in STATE_RECIPES:STATE_RECIPES[key]=STATE_RECIPES[context]
    else:RECIPES[key]=RECIPES[context]
RECIPES.update({
    "counter.sequence_operations":("([0.25]+[0.5])[1]",.5),
    "counter.short_circuits":("0.5 or 0.25",.5),
    "counter.sorted_calls":("sorted([0.75,0.5])[0]",.5),
    "counter.sort_key_calls":("sorted([0.25,0.5],key=lambda v:-v)[0]",.5),
})
# Parameter binding witnesses perturb the relevant bound values, not the body result.
BINDING_RECIPES={
    "signature.posonly":("def f(v,/,bias=0.25,*,gain=1):\n    return v*gain+bias\ncoverage_bias=f(0.25,gain=1)",.5),
    "signature.kwonly":("def f(v=0.25,/,*,gain=1):\n    return v*gain+0.25\ncoverage_bias=f(gain=1)",.5),
    "signature.vararg":("def f(*values,**options):\n    return values[0]+options['bias']\ncoverage_bias=f(*[0.25],**{'bias':0.25})",.5),
    "signature.kwarg":("def f(*values,**options):\n    return values[0]+options['bias']\ncoverage_bias=f(*[0.25],**{'bias':0.25})",.5),
    "expansion.star":("def f(*values,**options):\n    return values[0]+options['bias']\ncoverage_bias=f(*[0.25],**{'bias':0.25})",.5),
    "expansion.kwstar":("def f(*values,**options):\n    return values[0]+options['bias']\ncoverage_bias=f(*[0.25],**{'bias':0.25})",.5),
}
STATE_RECIPES.update(BINDING_RECIPES)
LAMBDA_RECIPES={
    'lambda.posonly':('f=lambda v,/,bias=0.25:v+bias\ncoverage_value=f(covered_input)-0.25',None),
    'lambda.kwonly':('f=lambda *,v:v+0.25\ncoverage_value=f(v=covered_input)-0.25',None),
    'lambda.vararg':('f=lambda *values:values[0]+values[1]\ncoverage_value=f(*(covered_input,0.25))-0.25',None),
    'lambda.kwarg':("f=lambda **values:values['value']+values['bias']\ncoverage_value=f(**{'value':covered_input,'bias':0.25})-0.25",None),
}
STATE_RECIPES.update(LAMBDA_RECIPES)

UNARY_RECIPES={
    "operator.UAdd":("a=np.array([0.5],dtype=object)\nb=+a\na[0]=0.75\ncoverage_bias=b[0]",.5),
    "call.UAdd":("a=np.array([0.5],dtype=object)\nb=np.positive(a)\ncoverage_bias=b[0]",.5),
    "operator.USub":("a=np.array([covered_input+0.25],dtype=object)\nb=-a\ncoverage_value=-b[0]-0.25",None),
    "call.USub":("a=np.array([covered_input+0.25],dtype=object)\nb=np.negative(a)\ncoverage_value=-b[0]-0.25",None),
    "cipher_cells":("a=np.array([covered_input+0.25],dtype=object)\nb=-a\ncoverage_value=-b[0]-0.25",None),
    "plain_cells":("p=Empty()+0.5\na=np.array([p],dtype=object)\nb=-a\ncoverage_value=covered_input+b[0]+0.5",None),
    "public_cells":("a=np.array([0.5],dtype=object)\nb=+a\na[0]=0.75\ncoverage_bias=b[0]",.5),
    "boolean_cells":("a=np.array([covered_input+0.25,0.5,True],dtype=object)\nb=-a\ncoverage_value=-b[0]+b[1]+b[2]+1.25",None),
    "fresh.UAdd":("a=np.array([0.5],dtype=object)\nb=+a\na[0]=0.75\ncoverage_bias=b[0]",.5),
    "fresh.USub":("a=np.array([covered_input+0.25],dtype=object)\nview=a[:1]\nb=-view\ncoverage_value=b[0]+view[0]*2-0.25",None),
    "input_view":("a=np.array([covered_input+0.25],dtype=object)\nview=a[:1]\nb=-view\ncoverage_value=b[0]+view[0]*2-0.25",None),
}
STATE_RECIPES.update(UNARY_RECIPES)
ARITHMETIC_RECIPES={
    "cipher_pair":("a=np.array([covered_input],dtype=object)\nb=np.array([zero_ct+1],dtype=object)\nc=a*b\ncoverage_value=c[0]",None),
    "empty_left":("a=np.array([Empty()],dtype=object)\na-=np.array([covered_input+0.25],dtype=object)\ncoverage_value=a[0]-0.25",None),
    "inplace.Sub":("a=np.array([Empty()],dtype=object)\na-=np.array([covered_input+0.25],dtype=object)\ncoverage_value=a[0]-0.25",None),
    "inplace.Mult":("a=np.array([covered_input+0.25],dtype=object)\nalias=a\na*=0.5\ncoverage_value=alias[0]*2-0.25",None),
    "inplace.Add":("a=np.array([zero_ct+0.25,zero_ct+0.5,covered_input],dtype=object)\na[1:]+=a[:-1]\ncoverage_value=a[2]-0.5",None),
    "overlap":("a=np.array([zero_ct+0.25,zero_ct+0.5,covered_input],dtype=object)\na[1:]+=a[:-1]\ncoverage_value=a[2]-0.5",None),
    "object.overlap_mult":("a=np.array([zero_ct+0.5,zero_ct+1,covered_input+2],dtype=object)\na[1:]*=a[:-1]\ncoverage_value=a[2]-2",None),
    "rank_broadcast":("a=np.array([[covered_input*0.5],[covered_input*0.5]],dtype=object)\nb=np.array([0.25,0.75])\nc=a+b\ncoverage_value=c[0,0]+c[1,1]-1",None),
    "zero_dim":("a=np.array(covered_input,dtype=object)\nb=np.array(0.5,dtype=object)\nc=a+b\ncoverage_value=c-0.5",None),
}
STATE_RECIPES.update(ARITHMETIC_RECIPES)
STORAGE_RECIPES={
    "attr.double":("coverage_bias=np.array([0.5],dtype=np.double)[0]",.5),
    "attr.float64":("coverage_bias=np.asarray([0.5],dtype=np.float64)[0]",.5),
    "call.empty":("a=np.empty(2,dtype=object)\nfor i in range(len(a)):\n    a[i]=zero_ct+0.25\ntotal=a[0]\nfor i in range(1,len(a)):\n    total+=a[i]\ncoverage_value=covered_input+total-0.5",None),
    "public.loop":("coverage_bias=0\nfor i in range(2):\n    coverage_bias+=0.25",.5),
}
EVIDENCE_SCOPES={"attr.double":"typed_float64_constructor_output",
                 "attr.float64":"typed_float64_constructor_output",
                 "call.empty":"initialized_object_allocation_shape"}
STATE_RECIPES.update(STORAGE_RECIPES)
VIEW_RECIPES={
    'storage.copy_independence':('a=np.array([covered_input+0.25,zero_ct+0.5],dtype=object)\nb=a.copy()\na[0]=a[0]+0.5\ncoverage_value=b[0]-0.25',None),
    'storage.reshape_view':('a=np.array([[covered_input+0.25,zero_ct+0.5]],dtype=object)\nb=a.reshape(2)\na[0,0]=a[0,0]+0.5\ncoverage_value=b[0]-0.75',None),
    'storage.transpose_view':('a=np.array([[covered_input+0.25,zero_ct+0.5]],dtype=object)\nb=a.transpose()\na[0,0]=a[0,0]+0.5\ncoverage_value=b[0,0]-0.75',None),
}
STATE_RECIPES.update(VIEW_RECIPES)
EVIDENCE_SCOPES.update({f:'object_storage_alias_distinction' for f in VIEW_RECIPES})
COMPOSITES={
    'array.arithmetic':('cipher_pair','empty_left','rank_broadcast','zero_dim'),
    'array.mutation':('inplace.Add','inplace.Sub','inplace.Mult','overlap','object.overlap_mult'),
    'array.view_copy':tuple(VIEW_RECIPES),
}
STRUCTURAL={"node.Pass"}
SPECS = {"unified-public-"+f.replace(".","-"):
         (f, "Use an executed "+f+" expression whose result contributes to a named output. "
          "Unused expressions, dead branches, cancelled values and unobserved slots do not qualify. "
          "Acceptance uses bounded finite public interventions at an executed source span; "
          "it is not an all-input or per-invocation proof.")
         for f in sorted(set(RECIPES)|set(STATE_RECIPES))}

for key,(feature,instruction) in list(SPECS.items()):
    if feature in STRUCTURAL:
        SPECS[key]=(feature,"Execute a reachable pass statement. This is structure-only evidence, with no numerical contribution claim.")
    elif feature.startswith("counter."):
        SPECS[key]=(feature,"Observe a positive trusted "+feature+" increment at its executed source span, and show finite output influence of the associated operation at that span. This is scoped operational evidence, not an assertion that the accounting integer itself is a numerical model input.")
    elif feature in STATE_RECIPES:
        SPECS[key]=(feature,"Execute "+feature+" with a matching control or state-change intervention that changes a named output. Preserve argument/RHS evaluation where the intervention suppresses a write. Dead/unused operations do not qualify; finite evidence is not an all-input proof.")
    if feature in TYPED_RULES:
        SPECS[key]=(feature,TYPED_RULES[feature]+" Require matching typed execution facts. "+instruction)
    if feature in STORAGE_RECIPES:
        if feature.startswith('attr.'):
            instruction="Use an executed numeric np.array/asarray constructor with direct dtype=np."+feature[5:]+" and consume its result in a named output. This proves the float64 type context and numerical contribution of the constructor result, not a numerical difference between the equivalent double/float64 aliases. Merely reading an unused dtype marker does not qualify."
        elif feature=='call.empty':
            instruction="Use a nonempty positive-rank np.empty(...,dtype=object), initialize cells before arithmetic, and consume the storage so that a bounded increase in its first dimension changes a named output. This is an allocation-shape witness, not invented numeric values for uninitialized None cells. Dead allocations or shape-independent fixed writes do not qualify."
        else:
            instruction="Execute a bounded public for loop with at least one actual iteration and a loop-body contribution to a named output. Shortening the iterable must change that output; empty/unused loops and iterator-only side effects do not qualify. The loop is construction-time public control, never encrypted data-dependent control."
        SPECS[key]=(feature,instruction+" Acceptance uses typed/operational tracing plus bounded finite output probes, not an all-input proof.")
    if feature in ARITHMETIC_RECIPES:
        detail={"cipher_pair":"Use two object arrays with actual broadcast-aligned Cipher times Cipher cells, not a scalar Cipher RHS or Cipher cells in disjoint operand positions; their product results must contribute.",
                "empty_left":"Use actual Empty-left subtraction whose identity semantics differ observably from replacing Empty by public zero.",
                "inplace.Mult":"Use name-target whole-array *= with an alias whose observed value distinguishes in-place mutation from fresh-array rebinding.",
                "inplace.Add":"Use actual overlapping slice += whose output distinguishes snapshot reads from sequential write-through reads.",
                "overlap":"Use overlapping in-place object-array views whose snapshot semantics are observable in a named output.",
                "object.overlap_mult":"Use overlapping slice *= whose output distinguishes snapshot reads from sequential write-through reads.",
                "inplace.Sub":"Use actual object-array -= and consume its result.",
                "rank_broadcast":"Use genuinely expanded rank-two-or-higher object broadcasting and consume the result.",
                "zero_dim":"Use a rank-zero object binary operation returning a scalar Expr and consume it."}[feature]
        SPECS[key]=(feature,detail+" Require typed executed facts and a matching bounded output intervention. Unused matching cells, disjoint arrays and unobserved alias/overlap behavior do not qualify. Finite evidence is not a per-cell, per-invocation or all-input proof.")
    if feature in UNARY_RECIPES:
        SPECS[key]=(feature,"Execute an actual typed object-array unary operation satisfying "+feature+". Cell-type requirements perturb only results from that type; freshness requirements replace fresh storage with an alias (and in-place negation for minus). The relevant typed results or storage distinction must affect a named output. Wrong types, unused cells, dead operations and irrelevant source mutations do not qualify. Finite probes are not per-cell, per-invocation or all-input proofs.")
    if feature in BINDING_RECIPES:
        SPECS[key]=(feature,"Use a named def construction helper (lambda signature witnesses are not yet bound). Execute "+feature+" and use its bound parameter or expanded values in a named output. The witness perturbs those values only, preserving argument evaluation; an unused/overwritten parameter, ignored/empty expansion, body-only result or argument side effect does not qualify. Finite evidence applies to at least one value in the parameter/expansion at this span, not every argument or invocation.")

# Old named-def instruction text remains hash-compatible. The lambda family is
# explicitly separate; its witnesses do not silently satisfy signature.* tasks.
for feature in LAMBDA_RECIPES:
    name='unified-public-'+feature.replace('.','-')
    SPECS[name]=(feature,'Execute a lambda with '+feature+' binding whose bound values contribute to a named output. '
        'Only bound values are perturbed, preserving default and argument evaluation; unused lambdas/parameters, '
        'empty expansions, length/key-only use, argument-only side effects and body-only changes do not qualify. '
        'Variadic probes preserve the outer tuple/mapping type, length and keys; probes use uniform/nonuniform offsets or scaling of bound values. '
        'Named def signatures are separate exercises. Finite evidence is not a claim about every argument or invocation.')
for feature in VIEW_RECIPES:
    name='unified-public-'+feature.replace('.','-')
    SPECS[name]=(feature,'Execute an object-array '+feature+' operation and observe the storage distinction in a named output. '
        'Actual copy storage must be independent; reshape/transpose must actually share storage. '
        'Replacing copy by alias, or a view by detached storage, must change that output. '
        'Merely consuming equal copied values without an observable storage distinction does not qualify. '
        'Noncontiguous reshape that allocates a copy cannot claim a reshape view. '
        'The probe retains argument/receiver evaluation and shape; finite evidence is not all-input equivalence.')
for feature,children in COMPOSITES.items():
    SPECS['unified-public-composite-'+feature.replace('.','-')]=(feature,
        'Satisfy ALL registered subrequirements in the same candidate: '+', '.join(children)+
        '. Each child requires its own matching executed typed facts, bounded intervention and output contribution. '
        'No credit for dead/unused operations, irrelevant storage changes, or different programs satisfying different children. '
        'Object cells are complete Exprs or allowed public cells, not ciphertext slots. '
        'This aggregate covers its explicit children, not every program or all possible NumPy operations.')


def compose_golden(source,children,request):
    """Manual parallel identity branches; all child witnesses must contribute."""
    tree=ast.parse(source);fn=next(n for n in tree.body if n.name=='golden');ret=fn.body[-1]
    require(type(ret) is ast.Return and type(ret.value) is ast.List,'Composite public output ABI')
    occupied={n.id for n in ast.walk(tree) if type(n) is ast.Name}
    occupied|={n.arg for n in ast.walk(tree) if type(n) is ast.arg}
    root='public_composite_input'
    while root in occupied:root='c_'+root
    occupied.add(root)
    body=[ast.Assign(targets=[ast.Name(id=root,ctx=ast.Store())],value=ret.value.elts[0])];outputs=[]
    for index,feature in enumerate(children):
        name=next(k for k,v in SPECS.items() if v[0]==feature)
        fragment=ast.parse(golden_variant('@hc.func("c")\ndef golden(fragment_input):\n return [fragment_input]\n',name,request)).body[0]
        prefix='publicpart'+str(index)+'_'
        while any(n.startswith(prefix) for n in occupied):prefix='c_'+prefix
        bound={n.id for n in ast.walk(fragment) if type(n) is ast.Name and type(n.ctx) is ast.Store}
        mapping={n:prefix+n for n in bound}
        class Rename(ast.NodeTransformer):
            def visit_Name(self,n):
                if n.id=='fragment_input':n.id=root
                elif n.id in mapping:n.id=mapping[n.id]
                return n
        Rename().visit(fragment);body.extend(fragment.body[:-1]);outputs.append(fragment.body[-1].value.elts[0])
        occupied.update(mapping.values())
    require(outputs,'Empty public aggregate')
    combined=outputs[0]
    for value in outputs[1:]:combined=ast.BinOp(left=combined,op=ast.Add(),right=value)
    fn.body[-1:-1]=body;ret.value.elts[0]=ast.BinOp(left=combined,op=ast.Mult(),right=ast.Constant(1/len(outputs)))
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def spec(name):
    require(name in SPECS, "Unknown unified public directed requirement")
    feature,instruction=SPECS[name]
    value=dict(id=name,required_features=list(COMPOSITES.get(feature,(feature,))),instruction=instruction)
    if feature in COMPOSITES:value["aggregate_requirement"]=feature
    return value

def validate_request(request):
    value=request.get("construction_exercise")
    require(type(value) is dict and value==spec(value.get("id")), "Changed public construction metadata")

def golden_variant(source,name,request=None):
    """Manual witness for any logical graph; not an Agent response."""
    spec(name);feature=SPECS[name][0]
    if feature in COMPOSITES:return compose_golden(source,COMPOSITES[feature],request)
    expression,value=(RECIPES.get(feature) or STATE_RECIPES[feature])
    tree=ast.parse(source);fn=next(n for n in tree.body if n.name=="golden");ret=fn.body[-1]
    require(type(ret) is ast.Return and type(ret.value) is ast.List,"Unexpected rule output")
    first=ast.unparse(ret.value.elts[0])
    if feature=="object.named_numeric_constructor":
        require(request is not None,"Named original constant witness needs its immutable request")
        half=next(k for k,v in request["public_constants"].items() if type(v) in (int,float) and v==.5)
        expression=expression.replace("SUPPLIED_HALF",half)
    if feature in STATE_RECIPES and value is None:
        body="covered_input = "+first+"\n"+expression+"\ncovered_out = coverage_value\n"
    elif feature in STATE_RECIPES:
        body=expression+"\ncovered_out = ("+first+" + coverage_bias) - "+repr(value)+"\n"
    elif value is None:
        body="covered_input = "+first+"\ncovered_out = "+expression+"\n"
    else:
        body="coverage_bias = "+expression+"\ncovered_out = ("+first+" + coverage_bias) - "+repr(value)+"\n"
    fn.body[-1:-1]=ast.parse(body).body
    ret.value.elts[0]=ast.Name(id="covered_out",ctx=ast.Load())
    return ast.unparse(ast.fix_missing_locations(tree))+"\n"
