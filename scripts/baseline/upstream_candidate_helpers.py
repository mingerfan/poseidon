"""Opt-in trusted upstream calls. Candidate Python stays inert, not imported.

Actual invocation evidence does not by itself establish returned contribution.
The first capability exposes only the fixed polynomial HE_SiLU, not exact SiLU.
"""
import hashlib
import json
from pathlib import Path
from benchmark_graph import digest, require, canonical

PROFILE = "upstream-poly-silu-v1"
BN_PROFILE = "upstream-poly-bn-silu-v2"
CONCAT_PROFILE = "upstream-poly-concat-bn-silu-v3"
SPATIAL_PROFILE = "upstream-poly-spatial-v4"
MAPPED_PROFILE = "upstream-poly-spatial-mapped-v5"
FUSED_PROFILE = "upstream-poly-fused-spatial-v6"
DS_PROFILE = "upstream-poly-downsample-v7"
VR_PROFILE = "upstream-poly-virtual-prefix-v8"
CHUNK_PROFILE = "upstream-poly-chunked-virtual-v9"
POLYNOMIAL_PROFILE = "upstream-poly-fixed-polynomials-v10"
POLYNOMIAL_RULES = """Optional fixed polynomial capability. The callees and exact Chebyshev coefficients
are listed in upstream_helpers.helpers. Each takes one whole ciphertext Expr and
returns one Expr via the unchanged source-locked upstream polynomial closure.
Poly_Default uses Poly.GenPoly(); the Tree and Leaf calls use MPCB.GenPoly.
Only the recorded odd leaf coefficients are evaluated by the pinned library.
These are polynomial functions, not exact sign, sigmoid or ReLU. No imports,
custom coefficients/trees, runtime closure construction or bootstrap are allowed.
Use the explicit function whose polynomial matches the intended graph. Numerical
acceptance still applies to the whole composition; a call is not proof of output
contribution. Native rules and resource budgets remain in force.
"""
LOCK = Path(__file__).with_name("upstream-helpers-v1.json")
SPECS = {"HE_SiLU": {"parameters": ["c"], "result": "c", "work": 256,
    "semantics": "fixed upstream tree/odd-leaf Chebyshev polynomial times x; not exact SiLU",
    "rotations": [], "bootstrap": False}}
RULES = """Optional trusted upstream capability: HE_SiLU(cipher_expr).
HE_SiLU calls the fixed upstream poly.Func.HE_SiLU unchanged. It computes a fixed
polynomial approximation, NOT exact SiLU. It accepts one whole ciphertext Expr,
returns one Expr and reserves 256 work units per call within existing budgets.
No imports, helper replacement, arbitrary poly functions, bootstrap, custom
coefficients or closure parameters are allowed. No helper is required in free
synthesis. Ordinary native rules still apply. A traced call alone is not proof
that it contributes to an output, and this capability is not a directed witness.
"""

BN_RULES = """Optional trusted upstream capability: HE_SiLU(cipher_expr), and the
HE_BN<number>(cipher_expr) callees listed in upstream_helpers.helpers. HE_SiLU is
the fixed polynomial approximation, not exact SiLU. Each HE_BN callee uses the
immutable public parameters and checked closure geometry of the named model node.
These calls accept/return whole ciphertext Exprs, whose C-order prefixes represent
the registered shapes; unused slots are unspecified. The candidate owns correct
input/layout composition. Each BN call reserves 128 work units; SiLU reserves 256.
The actual upstream HE_BN -> abstractBN -> shapeClosure.BN is called unchanged.
No arbitrary imports/helpers/closure construction, parameter replacement, bootstrap
or security changes are permitted. Unsupported bindings remain explicitly listed.
No helper is required in free synthesis. Invocation records alone do not establish
output contribution or directed coverage. Native construction limits still apply.
"""

CONCAT_RULES = BN_RULES + """Version 3 also exposes bound HE_Concat<number>(left,right) calls.
Each calls actual poly.Func.HE_Concat and its checked channel-concat closure.
Bindings require two equal-shape values: rank-one vectors, or batch-one rank2-4
channel tensors. Whole Expr inputs/output use their C-order prefix in common P.
No arbitrary closure or shape is accepted. The trusted right-input rotation
adapter composes signed rotations modulo P from existing positive power-of-two
steps, exactly for the invariant P-periodic slots. It adds no keys or permissions.
Each concat reserves 128 work units. Unsupported bindings remain in the request.
"""

SPATIAL_RULES = CONCAT_RULES + """Version 4 adds immutable bound HE_Conv<number>,
HE_Avg<number> and HE_Pool<number> calls from the listed graph nodes.
Each invokes the real upstream function and shapeClosure at nt=P. This nt is a
logical periodic ring, not the physical SEAL degree/slot count. A trusted Expr
adapter composes signed rotations modulo P into existing positive binary steps.
It checks all helper constants, sanitizes/packs input prefixes, and canonicalizes
output positions; upstream algorithms and physical security parameters are unchanged.
Shapes, weights, closure, work reservation and output positions are immutable.
Supported batch-one rank4 spatial inputs have power-of-two H/W, fit P and the
recorded output packing. Unsupported groups/dilation/stride/padding/geometry remain
listed with reasons. HE_Pool also binds spatial mean nodes, not arbitrary reductions.
The candidate receives only typed callees, never adapter classes or closure objects.
"""

MAPPED_RULES = CONCAT_RULES + """Version 5 exposes immutable HE_Conv<number>, HE_Avg<number>,
HE_Pool<number> bindings. The listed binding either uses the v4 exact periodic
ring, or fixed public gather/scatter maps around one or more real upstream calls.
Mapped Conv adds only zero coefficients for even kernels, groups and dilation,
with effective kernels at most 3; fixed tiles select exactly the original windows.
Mapped 2x2 Avg uses actual HE_Avg per window, including the original padding divisor.
Mapped spatial mean uses actual HE_Pool plus the exact public normalization factor.
Every inner closure, parameter, mask, rotation, output index and total work is
immutable, hash-bound, and limited by the unchanged 1024 golden-work budget.
Exact public-zero intermediates inside mapped calls are elided and recorded;
public-only final results require an explicit encrypted-zero lowering and remain
blocked here. No encrypted value is inspected and no near-zero threshold is used.
C-order logical prefixes, physical slots, security parameters and rotation keys
are unchanged. Candidates cannot construct adapters or choose mappings. Unsupported
bindings remain explicit; no arbitrary helper, bootstrap or chunk mixing is added.
"""

FUSED_RULES = MAPPED_RULES + """Version 6 additionally exposes HE_ConvBN<number> and
HE_DwConv<number>. Each immutable binding spans a Conv2d -> BatchNorm edge,
accepts the pre-convolution value and returns the BN output in canonical C-order.
The actual upstream fused function runs unchanged. Conv bias is represented
exactly by a public BN mean shift; the actual abstractBN derives gain and offset.
DwConv requires one output filter per input channel and a stride-one inner closure.
Raw Conv and BN bindings remain separately available for fanout. The candidate
must choose the correct bound input, not feed the Conv output into a fused call.
All derived parameters, sparse maps and work are fixed and checked. No new keys,
bootstrap, arbitrary closures, chunk mixing or security relaxation is allowed.
"""

DS_RULES = FUSED_RULES + """Version 7 adds bound HE_DS<number> callees for a producer
chain of two static step-two spatial slices on distinct H/W axes, in either order.
The input is the value before both slices; the output is after both slices.
The binding keeps each original start/stop and shape, including odd or singleton
dimensions, using fixed full input tiles around actual HE_DS and its original
DownSelecting/Downsamp closure. Actual centering/duplication is mapped back to the
canonical prefix. This is not a direct slice replacement of the requested helper.
Every tile, closure, sparse map and work reservation is fixed and hash-checked.
Existing physical slots, keys, budgets, isolation and helper restrictions remain.
"""


VR_RULES = DS_RULES + """Version 8 adds bound HE_MPBN<number>, HE_Linear<number>,
and HE_ReshapeLinear<number>(cipher_expr, zero_ct). Both arguments are ciphertext
Exprs; the second is the declared encrypted-zero input, used for public-only
results. The first is the bound input_value (pre-flatten for ReshapeLinear).
A sparse virtual 65536-slot ring preserves every logical slot and full upstream
constant until the final declared output-prefix projection. Physical slots stay
16384, each live block is P-periodic, and existing binary rotations are reused.
MPBN expands fixed channel parameters in C order; Linear may add exact zero
columns, process fixed leading rows, and apply fixed gather/scatter maps.
ReshapeLinear invokes actual MPCB.Reshape and Linear, with matching input
permutation; it is never replaced by a direct linear lowering. Public padding,
maps, work, shape and parameters are immutable. Unknown encrypted values are
never inspected. At most 64 live virtual blocks/calls and 1024 work are allowed.
Actual full-constant hashes and complete slot-support transcripts are checked
against an independent public-structure replay, separate from the numerical
reference. Unsupported bindings stay listed. No bootstrap or chunk mixing.
"""

CHUNK_RULES = """Optional version 9 chunked upstream capability.
Only HE_SiLU and the bound HE_MPBN/HE_Linear/HE_ReshapeLinear callees listed in
upstream_helpers.helpers are available. A bound callee accepts every P-periodic
chunk of its input_value in C-order offset order, followed by the declared zero_ct.
It returns exactly its bound output_chunk, repeated every P slots. Different
callee suffixes select different chunks of the same full mathematical output.
The original upstream helper operates on a complete sparse 65536-slot virtual
ring assembled from all input chunks. No chunk is treated as an independent
whole tensor. ReshapeLinear preserves its original Reshape and Linear calls.
Shapes, coefficients, full constants, mapping, output selection and work are
immutable and audited. Public maps may use a larger finite index space Q; Q is
not a physical slot period or permission for additional rotation keys.
At most four input chunks and four output chunks, 256 logical elements,
64 live virtual blocks, and the existing 1024 total native work budget apply.
A call for each output chunk performs the full original computation; work is
charged per call. Output contribution is tested separately from call presence.
No arbitrary helper/closure imports, parameter replacement, bootstrap, security
changes or nonperiodic-constant truncation. Unsupported bindings stay listed.
Other helper families remain unavailable in this version; old profiles and
periodic requests retain their existing contracts.
"""

def manifest(model=None,profile=PROFILE,chunk_period=None):
    if profile==POLYNOMIAL_PROFILE:
        require(chunk_period is None,"Fixed polynomial capability requires periodic layout")
        from upstream_adapters.fixed_polynomial import specs
        return dict(profile=profile,helpers=specs(),
                    source_lock_sha256=hashlib.sha256(LOCK.read_bytes()).hexdigest())
    if profile==CHUNK_PROFILE:
        from unified_chunk_layout import layout as chunk_layout
        from upstream_adapters.chunk_virtual_node import bind_node,FAMILIES
        from benchmark_graph import validate
        import math
        plan=chunk_layout(model,chunk_period);checked=validate(model)
        helpers=json.loads(json.dumps(SPECS));blocked=[]
        for family in FAMILIES:
            for index,node in enumerate(n for n in model["nodes"] if n["op"]==("batch_norm" if family=="HE_MPBN" else "linear")):
                count=math.prod(checked["shapes"][node["outputs"][0]])
                try:
                    specs={}
                    for output in range((count+chunk_period-1)//chunk_period):
                        b=bind_node(model,node["id"],chunk_period,family,output)
                        name=family+str(index)+"_chunk"+str(output)
                        specs[name]=dict(parameters=["c"]*(b["input_chunks"]+1),result="c",work=b["work"],
                            rotations=[1<<j for j in range(chunk_period.bit_length()-1)],bootstrap=False,
                            semantics="actual upstream helper over all logical input chunks; fixed output projection",binding=b)
                    helpers.update(specs)
                except ValueError as error:
                    blocked.append(dict(node_id=node["id"],helper=family,reason=str(error)))
        return dict(profile=profile,helpers=helpers,source_lock_sha256=hashlib.sha256(LOCK.read_bytes()).hexdigest(),
                    chunk_period=plan["input_slot_period"],unavailable_bindings=blocked)
    require(chunk_period is None,"Chunk period requires the chunked helper profile")

    require(profile in (PROFILE,BN_PROFILE,CONCAT_PROFILE,SPATIAL_PROFILE,MAPPED_PROFILE,FUSED_PROFILE,DS_PROFILE,VR_PROFILE),"Unknown upstream profile")
    helpers=json.loads(json.dumps(SPECS))
    body=dict(profile=profile,helpers=helpers,source_lock_sha256=hashlib.sha256(LOCK.read_bytes()).hexdigest())
    if profile in (BN_PROFILE,CONCAT_PROFILE,SPATIAL_PROFILE,MAPPED_PROFILE,FUSED_PROFILE,DS_PROFILE,VR_PROFILE):
        from unified_graph_contract import layout
        from upstream_adapters.batch_norm_node import bind_node
        period=layout(model)['input_slot_period'];blocked=[]
        for i,node in enumerate(n for n in model['nodes'] if n['op']=='batch_norm'):
            try:binding=bind_node(model,node['id'],period)
            except ValueError as error:
                blocked.append(dict(node_id=node['id'],reason=str(error)));continue
            helpers['HE_BN'+str(i)]=dict(parameters=['c'],result='c',work=128,rotations=[],bootstrap=False,
                semantics='actual upstream BatchNorm with immutable public parameters',binding=binding)
        if profile in (CONCAT_PROFILE,SPATIAL_PROFILE,MAPPED_PROFILE,FUSED_PROFILE,DS_PROFILE,VR_PROFILE):
            from upstream_adapters.concat_node import bind_node as bind_concat
            for i,node in enumerate(n for n in model['nodes'] if n['op']=='concat'):
                try:binding=bind_concat(model,node['id'],period)
                except ValueError as error:
                    blocked.append(dict(node_id=node['id'],helper='HE_Concat',reason=str(error)));continue
                helpers['HE_Concat'+str(i)]=dict(parameters=['c','c'],result='c',work=128,
                    rotations=binding['rotation']['steps'],bootstrap=False,
                    semantics='actual upstream channel concat with immutable geometry',binding=binding)
        if profile in (SPATIAL_PROFILE,MAPPED_PROFILE,FUSED_PROFILE,DS_PROFILE,VR_PROFILE):
            if profile in (MAPPED_PROFILE,FUSED_PROFILE,DS_PROFILE,VR_PROFILE):
                from upstream_adapters.spatial_mapped import bind_node as bind_spatial
            else:
                from upstream_adapters.spatial_node import bind_node as bind_spatial
            counts={'HE_Conv':0,'HE_Avg':0,'HE_Pool':0}
            for node in model['nodes']:
                if node['op'] not in ('conv2d','avg_pool2d','mean'):continue
                try:binding=bind_spatial(model,node['id'],period)
                except ValueError as error:
                    blocked.append(dict(node_id=node['id'],helper='spatial',reason=str(error)));continue
                family=binding['helper'];name=family+str(counts[family]);counts[family]+=1
                helpers[name]=dict(parameters=['c'],result='c',work=binding['work'],
                    rotations=[1<<j for j in range(period.bit_length()-1)],bootstrap=False,
                    semantics='actual upstream spatial helper in exact periodic ring',binding=binding)
        if profile in (FUSED_PROFILE,DS_PROFILE,VR_PROFILE):
            from upstream_adapters.fused_conv_bn import bind_node as bind_fused
            for i,node in enumerate(n for n in model['nodes'] if n['op']=='batch_norm'):
                for family in ('HE_ConvBN','HE_DwConv'):
                    try:binding=bind_fused(model,node['id'],period,family)
                    except ValueError as error:
                        blocked.append(dict(node_id=node['id'],helper=family,reason=str(error)));continue
                    helpers[family+str(i)]=dict(parameters=['c'],result='c',work=binding['work'],
                        rotations=[1<<j for j in range(period.bit_length()-1)],bootstrap=False,
                        semantics='actual upstream fused Conv and BN with immutable equivalent public parameters',binding=binding)
        if profile in (DS_PROFILE,VR_PROFILE):
            from upstream_adapters.downsample_node import bind_node as bind_ds
            i=0
            for node in model['nodes']:
                if node['op']!='slice':continue
                try:binding=bind_ds(model,node['id'],period)
                except ValueError as error:
                    blocked.append(dict(node_id=node['id'],helper='HE_DS',reason=str(error)));continue
                helpers['HE_DS'+str(i)]=dict(parameters=['c'],result='c',work=binding['work'],
                    rotations=[1<<j for j in range(period.bit_length()-1)],bootstrap=False,
                    semantics='actual upstream downsample with immutable spatial slice edge and public layout maps',binding=binding)
                i+=1
        if profile==VR_PROFILE:
            from upstream_adapters.virtual_node import bind_node as bind_virtual
            for family in ('HE_MPBN','HE_Linear','HE_ReshapeLinear'):
                for i,node in enumerate(n for n in model['nodes'] if n['op']==('batch_norm' if family=='HE_MPBN' else 'linear')):
                    try:binding=bind_virtual(model,node['id'],period,family)
                    except ValueError as error:
                        blocked.append(dict(node_id=node['id'],helper=family,reason=str(error)));continue
                    helpers[family+str(i)]=dict(parameters=['c','c'],result='c',work=binding['work'],
                        rotations=[1<<j for j in range(period.bit_length()-1)],bootstrap=False,
                        semantics='actual upstream helper on a checked full sparse virtual ring',binding=binding)
        body['unavailable_bindings']=blocked
    return body

def validate_manifest(value,model=None,layout=None):
    period=layout.get("input_slot_period") if layout and type(value) is dict and value.get("profile")==CHUNK_PROFILE else None
    require(type(value) is dict and canonical(value)==canonical(manifest(model,value.get('profile'),period)),
            "Changed upstream helper capability")

def dispatch_specs(request=None):
    if request is None:return SPECS
    from unified_graph_contract import validate_request
    validate_request(request)
    require('upstream_helpers' in request,'Missing upstream capability')
    return request['upstream_helpers']['helpers']

def verify_sources(root=None):
    lock=json.loads(LOCK.read_text())
    require(lock["schema"]==1 and lock["gitlink"]=="4616402710f39df3e5f5bd7930a6c036025aaac3",
            "Upstream helper source identity")
    if root is None:
        from workspace_paths import ROOT
        root=ROOT
    root=Path(root)
    prefix="third_party/dacapo/python/poly/"
    expected_poly={name for name in lock["sources"] if name.startswith(prefix)}
    actual_poly={str(p.relative_to(root)) for p in (root/prefix).rglob("*")
                 if p.is_file()}
    require(actual_poly==expected_poly,"Unexpected upstream helper source files")
    for name,expected in lock["sources"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe source lock path")
        path=root/name
        require(path.is_file() and not path.is_symlink() and
                hashlib.sha256(path.read_bytes()).hexdigest()==expected, "Changed helper source: "+name)
    path=root/"src/poseidon/tools/dacapo/poly-python-wheels.lock.json"
    require(hashlib.sha256(path.read_bytes()).hexdigest()==lock["dependency_lock_sha256"],
            "Changed helper dependency lock")
    return dict(source_lock_sha256=hashlib.sha256(LOCK.read_bytes()).hexdigest(),
                sources_sha256=digest(lock["sources"]),gitlink=lock["gitlink"],
                adapters={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((root/"scripts/baseline/upstream_adapters").glob("*.py"))})

def load(frontend,request=None,observe_binding=None):
    # The narrow poly mount has already passed host-side source/dependency checks.
    # Check its contents again before import in the network-isolated worker.
    import sys
    lock=json.loads(LOCK.read_text())
    prefix="third_party/dacapo/python/poly/"
    expected={name[len(prefix):]:value for name,value in lock["sources"].items() if name.startswith(prefix)}
    actual={str(p.relative_to("/upstream-poly")) for p in Path("/upstream-poly").rglob("*")
            if p.is_file()}
    require(actual==set(expected),"Unexpected upstream helper source files")
    for name,sha in expected.items():
        path=Path("/upstream-poly")/name
        require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==sha,
                "Changed sandbox helper source")
    sys.modules["hecate"]=frontend
    sys.path[:0]=["/poly-deps","/upstream-poly"]
    import einops
    require(einops.__version__=="0.6.1","Helper dependency version")
    import poly.Func as upstream
    if request is not None and request["upstream_helpers"]["profile"]==POLYNOMIAL_PROFILE:
        import poly.Poly as poly
        import poly.MPCB as mpcb
        from upstream_adapters.fixed_polynomial import load as load_polynomials
        return load_polynomials(poly,mpcb)
    # Explicit immutable dispatch; no getattr or candidate-supplied callable.
    calls={"HE_SiLU": upstream.HE_SiLU}
    specs=dispatch_specs(request)
    if request is not None and request["upstream_helpers"]["profile"]==CHUNK_PROFILE:
        from upstream_adapters.chunk_virtual_node import apply as apply_chunk
        def chunk_wrapper(name,binding):
            def invoke(*args):
                result,record=apply_chunk(binding,list(args[:-1]),args[-1],upstream)
                if observe_binding is not None:
                    observe_binding(dict(callee=name,binding_sha256=digest(binding),actual=record))
                return result
            return invoke
        for name,spec in specs.items():
            if name!="HE_SiLU":calls[name]=chunk_wrapper(name,spec["binding"])
        return calls
    if request is not None and request['upstream_helpers']['profile'] in (BN_PROFILE,CONCAT_PROFILE,SPATIAL_PROFILE,MAPPED_PROFILE,FUSED_PROFILE,DS_PROFILE,VR_PROFILE):
        import poly.MPCB as mpcb
        from upstream_adapters.batch_norm_node import apply
        def wrapper(name,binding):
            def invoke(cipher):
                result,record=apply(binding,cipher,upstream,mpcb)
                if observe_binding is not None:
                    observe_binding(dict(callee=name,binding_sha256=digest(binding),actual=record))
                return result
            return invoke
        def concat_wrapper(name,binding):
            from upstream_adapters.concat_node import apply as apply_concat
            def invoke(left,right):
                result,record=apply_concat(binding,left,right,upstream,mpcb)
                if observe_binding is not None:
                    observe_binding(dict(callee=name,binding_sha256=digest(binding),actual=record))
                return result
            return invoke
        def spatial_wrapper(name,binding):
            if binding.get('adapter')=='mapped-downsample-v1':
                from upstream_adapters.downsample_node import apply as apply_spatial
            elif binding.get('adapter')=='fused-conv-bn-v1':
                from upstream_adapters.fused_conv_bn import apply as apply_spatial
            else:
                from upstream_adapters.spatial_mapped import apply as apply_spatial
            def invoke(cipher):
                result,record=apply_spatial(binding,cipher,upstream,mpcb)
                if observe_binding is not None:
                    observe_binding(dict(callee=name,binding_sha256=digest(binding),actual=record))
                return result
            return invoke
        def virtual_wrapper(name,binding):
            from upstream_adapters.virtual_node import apply as apply_virtual
            def invoke(cipher,zero_cipher):
                result,record=apply_virtual(binding,cipher,zero_cipher,upstream,mpcb)
                if observe_binding is not None:
                    observe_binding(dict(callee=name,binding_sha256=digest(binding),actual=record))
                return result
            return invoke
        for name,spec in specs.items():
            if spec.get('binding',{}).get('adapter')=='virtual-prefix-v1':
                calls[name]=virtual_wrapper(name,spec['binding']);continue
            if name.startswith(('HE_Conv','HE_Avg','HE_Pool','HE_DwConv','HE_DS')):
                calls[name]=spatial_wrapper(name,spec['binding']);continue
            if name.startswith('HE_Concat'):calls[name]=concat_wrapper(name,spec['binding'])
            elif name!='HE_SiLU':calls[name]=wrapper(name,spec['binding'])
    return calls

def verify_events(value,request,source):
    validate_manifest(request["upstream_helpers"],request["model"],request["layout"])
    profile=request["upstream_helpers"]["profile"]
    bound=profile in (BN_PROFILE,CONCAT_PROFILE,SPATIAL_PROFILE,MAPPED_PROFILE,FUSED_PROFILE,DS_PROFILE,VR_PROFILE,CHUNK_PROFILE)
    specs=dispatch_specs(request)
    require(type(value) is dict and set(value)==({"schema","profile","request_id","source_sha256","calls","candidate_python_executed","output_contribution_proven"}|({"bound_calls"} if bound else set())),
            "Helper trace fields")
    require(value["schema"]==1 and value["profile"]==profile and value["request_id"]==request["request_id"] and
            value["source_sha256"]==hashlib.sha256(source.encode()).hexdigest() and
            value["candidate_python_executed"] is False and value["output_contribution_proven"] is False,
            "Helper trace identity")
    import ast
    tree=ast.parse(source)
    sites={(f.name,n.func.id,tuple((n.lineno,n.col_offset,n.end_lineno,n.end_col_offset)))
           for f in tree.body for n in ast.walk(f) if type(n) is ast.Call and type(n.func) is ast.Name and n.func.id in specs}
    # Public loops are expanded before native tracing: spans retain original AST positions.
    require(type(value["calls"]) is list and len(value["calls"])<=4096,"Helper call record budget")
    for c in value["calls"]:
        require(type(c) is dict and set(c)=={"caller","callee","span","actual_upstream"} and
                c["actual_upstream"] is True and type(c["span"]) is list and
                (c["caller"],c["callee"],tuple(c["span"])) in sites,"Unbound actual helper call")
    from unified_graph_contract import validate_candidate
    from collections import Counter
    checked=validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=source),request)
    key=lambda c:(c['caller'],c['callee'],tuple(c['span']))
    require(Counter(map(key,value['calls']))==Counter(map(key,checked['upstream_calls'])),
            "Missing or duplicated actual helper calls")
    if bound:
        records=value['bound_calls']
        require(type(records) is list and len(records)<=4096,'Bound helper record budget')
        expected=Counter(c['callee'] for c in value['calls'] if c['callee']!='HE_SiLU')
        require(Counter(c.get('callee') for c in records)==expected,'Bound helper invocation count')
        for record in records:
            name=record['callee'];binding=specs[name]['binding'];actual=record['actual']
            require(set(record)=={'callee','binding_sha256','actual'} and record['binding_sha256']==digest(binding),
                    'Bound helper identity')
            if binding.get("adapter")=="chunked-virtual-prefix-v1":
                from upstream_adapters.chunk_virtual_node import verify_record
                verify_record(binding,actual);continue
            if binding.get('adapter')=='virtual-prefix-v1':
                from upstream_adapters.virtual_node import verify_record
                verify_record(binding,actual);continue
            if name.startswith(('HE_Conv','HE_Avg','HE_Pool','HE_DwConv','HE_DS')):
                if binding.get('adapter')=='mapped-downsample-v1':
                    from upstream_adapters.downsample_node import verify_record
                elif binding.get('adapter')=='fused-conv-bn-v1':
                    from upstream_adapters.fused_conv_bn import verify_record
                else:
                    from upstream_adapters.spatial_mapped import verify_record
                verify_record(binding,actual)
                continue
            if name.startswith('HE_Concat'):
                from upstream_adapters.concat_node import expected_record as concat_record
                require(canonical(actual)==canonical(concat_record(binding)),'Changed actual concat closure/rotation record')
                continue
            require(actual['helper']=='HE_BN' and actual['source']=='poly/Func.py' and
                    actual['closure']=='MPCB.shapeClosure.BN' and actual['parameter_derivation']=='MPCB.abstractBN' and
                    actual['input_shape']==binding['input_shape'] and actual['closure_period']==binding['closure_period'] and
                    actual['slot_count']==16384 and actual['unchanged_upstream_functions'] is True and
                    actual['bootstrap_removed'] is False and actual['output_contribution_proven'] is False and
                    actual['returned_to_golden'] is False and actual['agent_generated'] is False,
                    'Actual BN closure identity')
            from upstream_adapters.batch_norm_node import expected_record
            require(canonical(actual)==canonical(expected_record(binding)), 'Changed actual BN closure record')
            geometry=actual['geometry']
            require(geometry['ni']==geometry['no']==geometry['ko']==1 and geometry['nt']==16384 and
                    geometry['nt']//geometry['po']==binding['closure_period'],'Actual BN physical geometry')
    result=dict(profile=profile,actual_upstream_calls=len(value["calls"]),
                output_contribution_proven=False,agent_generation_proven=False)
    if profile in (MAPPED_PROFILE,FUSED_PROFILE,DS_PROFILE,VR_PROFILE):
        result['actual_inner_spatial_calls']=sum(len(r['actual'].get('inner_calls',[r['actual']]))
            for r in value['bound_calls'] if r['callee'].startswith(('HE_Conv','HE_Avg','HE_Pool','HE_DwConv','HE_DS')))
    return result
