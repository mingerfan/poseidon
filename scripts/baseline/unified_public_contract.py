"""Explicit unified public-construction profile, separate from native Func calls."""
import ast
import hashlib
import json
from seal_artifact_gate import require

CONTRACT="hecate-unified-public-v1"
RULES="""Implement the immutable logical graph and all named outputs.
The trusted runner supplies hc and public_constants; no imports or candidate Python execution.
Only golden is decorated @hc.func with one c per ordered logical input plus zero_ct.
Other functions are undecorated construction-time AST helpers, not native Hecate Func calls.
Use bounded lexical helpers/closures/nonlocal, defaults, positional/keyword binding, lambdas,
list/tuple/dict containers and views, public comprehensions, public if/for/while,
break/continue and early helper returns. All conditions, indices and iteration are public.
Allowed public builtins: range,len,enumerate,zip,reversed,iter,next,list,tuple,dict,sorted,
int,float,pow. No eval/exec, arbitrary attributes, I/O, network, reflection or external calls.
Public strings allow strip/lstrip/rstrip,split/rsplit,partition/rpartition,replace,join.
Numeric np.array/asarray use finite bounded real data, reshape/flatten, shape and item.
Public np.floor/ceil/log2 and np.polynomial.Chebyshev with coef/domain/window are allowed.
Object arrays support the bounded registered np.empty/full/concatenate, indexing, copies,
views, reshape/transpose/T, object arithmetic and name-target mutation, np.negative/positive.
Empty()/hc.Empty() is only a symbolic construction identity; never encryption or bootstrap.
Whole Expr objects are array cells, not slots. Golden returns exactly one cipher Expr
per named output, in a flat list/tuple or a one-dimensional object array; no implicit flattening.
Cipher arithmetic is +,-,*,unary minus,rotate(k); each arithmetic pair requires a cipher.
Every input uses separate C-order padded packing repeated every P=layout.input_slot_period.
Outputs use the fixed C-order prefix and frozen selectors. Only positive power-of-two
rotations below P are legal. Compose rotations for other offsets.
Public supplied constants stay immutable. Derived Plain encoding is scalar or length P;
public arrays/containers remain limited to128 elements, rank<=4; shape rules still apply.
Finite public scalar magnitude<=1024 when encoded as Plain. No ciphertext conditions,
training, random forward, bootstrap, level changes or plaintext replacement of FHE.
At most17 total function/lambda declarations,128 helper calls,call depth<16;
source<=65536 bytes,AST<=4096 nodes,interpretation<=4096 steps,emitted cipher ops<=256.
Other existing type,integer,string,container,iteration,recursion and sandbox bounds apply.
Compiler owns scale/levels,rescale,modswitch,relinearization and security; client owns keys.
Do not alter model weights,layout,reference,test inputs,tolerance or compiler configuration.
Return only response-schema JSON. Acceptance requires real compilation and encrypted comparison.
"""
GUIDANCE={"schema":1,"profile":CONTRACT,"model":"logical tensor graph",
          "packing":"common periodic inputs, named packed outputs",
          "helper_semantics":"trusted bounded public AST construction, not native Func calls"}

def normalize(source,request,observe=None,*,_unary_probe=None,_object_probe=None,_storage_probe=None):
    from function_construction import normalize as expand
    from hecate_contract import UNIFIED_FLAT_CONTRACT
    names=tuple(x["dsl_name"] for x in request["layout"]["inputs"])+("zero_ct",)
    result=expand(source,request["public_constants"],request["layout"]["output_ciphertexts"],
                  input_names=names,object_unary=True,flat_contract=UNIFIED_FLAT_CONTRACT,
                  slot_period=request["layout"]["input_slot_period"],observe=observe,observe_counters=True,
                  _unary_probe=_unary_probe,_object_probe=_object_probe,_storage_probe=_storage_probe)
    require(all(result["constants"][k]==v for k,v in request["public_constants"].items()),
            "Construction changed supplied constants")
    return result

def validate(source,request,*,check_construction=True):
    from decorated_functions import validate as native_validate
    expanded=normalize(source,request)
    result=native_validate(expanded["source"],expanded["constants"],request["layout"]["output_ciphertexts"],
        input_names=tuple(x["dsl_name"] for x in request["layout"]["inputs"])+("zero_ct",),
        slot_period=request["layout"]["input_slot_period"])
    require(result["functions"]["golden"]["expanded_cost"]<=1024,"Unified candidate work budget")
    result["construction_profile"]=CONTRACT
    result["public_construction"]=expanded["construction"]
    if check_construction and "construction_exercise" in request:
        from unified_public_coverage import check_exercise
        result["construction_exercise"]=check_exercise(source,request)
    return result

def event_record(node):
    from construction_exercises import node_features
    facts={}
    if type(node) is tuple:
        require(len(node) in (2,3),"Unknown public event format")
        event,value,*detail=node
        facts=detail[0] if detail else {}
        require(type(facts) is dict,"Public event facts must be structured")
        node=value;features=["event."+event]
        if event=="resource_counter":
            require(set(facts)=={"counter","before","after"} and type(facts["counter"]) is str and
                    type(facts["before"]) is int and type(facts["after"]) is int and
                    0<=facts["before"]<facts["after"],"Counter event facts")
            features.append("counter."+facts["counter"])
            if facts['counter']=='loop_iterations' and type(node) is ast.For:
                features.append('public.loop')
        if event=='object_storage':
            require(facts['operation'] in ('copy','reshape','transpose') and type(facts['shares_storage']) is bool,
                    'Object storage facts')
            if facts['operation']=='copy' and not facts['shares_storage']:
                features.append('storage.copy_independence')
            elif facts['shares_storage']:
                features.append('storage.'+facts['operation']+'_view')
        if event=='numeric_constructor' and facts.get('direct_marker') in ('double','float64'):
            require(facts['floating'] and facts['explicit_float64'],'Typed dtype constructor facts')
            features.append('attr.'+facts['direct_marker'])
        if event=='object_allocation':
            require(facts['dtype']=='object' and facts['initial_cells']=='None','Object allocation facts')
            features.append('call.empty')
        if event in ("object_binary","object_inplace"):
            from object_arithmetic_exercises import features as arithmetic_features
            typed=arithmetic_features(event,node,facts)
            if type(node.op) is not ast.Mult:typed.discard('cipher_pair')
            features+=sorted(typed)
            if event=='object_inplace' and type(node.op) is ast.Mult and facts['overlapping']:
                features.append('object.overlap_mult')
        if event=="object_unary":
            from object_unary_exercises import features as unary_features
            features+=sorted(unary_features(facts)-{"zero_dim"})
            # Rank-zero unary returns a scalar, not newly allocated array storage.
            if facts['shape'] and sum(facts[k] for k in
                    ('cipher_cells','plain_cells','boolean_cells','public_real_cells'))>0:
                features.append('fresh.'+facts['operator'])
        if event in ("scalar_cast","scalar_item","object_numeric_constructor"):
            from scalar_conversion_exercises import features as typed_features
            features+=sorted(typed_features(event,facts))
    else:
        features=sorted(node_features(node))
        if type(node) is ast.Call and type(node.func) is ast.Name:
            builtins={"range","len","dict","Empty","enumerate","zip","reversed","iter","next","list","tuple","sorted","float","int","pow"}
            if node.func.id not in builtins:
                features=[f for f in features if f != "call."+node.func.id]
        if type(node) is ast.Lambda:
            for suffix,field in (('posonly','posonlyargs'),('kwonly','kwonlyargs'),('vararg','vararg'),('kwarg','kwarg')):
                if getattr(node.args,field):features.append('lambda.'+suffix)
    return dict(features=features,facts=facts,span=[getattr(node,k,0) for k in
                   ("lineno","col_offset","end_lineno","end_col_offset")])
