"""Explicit multi-ciphertext tensor binding within the existing four-I/O budget."""
import math
from benchmark_graph import require,dimensions,NAME,canonical,validate

ABI="unified-chunked-inputs-v1"
PERIODS=(4,8,16,32,64,128,256)

def declarations(inputs,outputs,period):
    require(type(period) is int and period in PERIODS,"Chunk period must be configured")
    require(type(inputs) is list and 1<=len(inputs)<=4 and type(outputs) is list and 1<=len(outputs)<=4,
            "Chunk logical I/O count")
    for entries in (inputs,outputs):
        require(all(type(s) is dict and set(s)=={"name","shape"} and type(s["name"]) is str
                    and NAME.fullmatch(s["name"]) for s in entries),"Chunk tensor declaration")
        require(len({s["name"] for s in entries})==len(entries),"Duplicate chunk tensor name")
        sizes=[math.prod(dimensions(s["shape"])) for s in entries]
        require(sum(sizes)<=256,"Chunk logical element budget")
    physical=[];results=[];selectors=[];ciphertext=0
    for spec in inputs:
        size=math.prod(spec["shape"])
        for start in range(0,size,period):
            require(len(physical)<4,"Chunk input ciphertext budget (maximum 4)")
            physical.append(dict(name=spec["name"],dsl_name=("x","y","z","t")[len(physical)],
                shape=spec["shape"],offset=start,elements=min(period,size-start)))
    for spec in outputs:
        size=math.prod(spec["shape"]);chunks=[]
        for start in range(0,size,period):
            require(ciphertext<4,"Chunk output ciphertext budget (maximum 4)")
            count=min(period,size-start)
            chunks.append(dict(ciphertext=ciphertext,offset=start,elements=count))
            selectors.extend([[ciphertext,i] for i in range(count)]);ciphertext+=1
        results.append(dict(name=spec["name"],shape=spec["shape"],chunks=chunks))
    return dict(execution_abi=ABI,input_slot_period=period,logical_inputs=inputs,inputs=physical,
        outputs=results,output_ciphertexts=ciphertext,output_selectors=selectors,
        output_shape=[len(selectors)],output_representation="named_chunked_prefixes",
        auxiliary_ciphertexts=[dict(dsl_name="zero_ct",kind="encrypted_zero",source="trusted_client",slot_period=period)])

def layout(model,period):
    checked=validate(model)
    return declarations(model["inputs"],[dict(name=o["name"],shape=checked["shapes"][o["value"]])
                                        for o in model["outputs"]],period)

def validate_layout(value):
    require(type(value) is dict and value.get("execution_abi")==ABI,"Chunk layout identity")
    require(type(value.get("outputs")) is list and all(type(o) is dict and {"name","shape"}<=set(o)
            for o in value["outputs"]),"Chunk output declarations")
    expected=declarations(value.get("logical_inputs"),
        [dict(name=o["name"],shape=o["shape"]) for o in value["outputs"]],value.get("input_slot_period"))
    require(canonical(value)==canonical(expected),"Changed chunk physical layout")
    return expected["input_slot_period"]

def chunk_rules(rules,public=False):
    if public:
        rules=rules.replace("one c per ordered logical input plus zero_ct",
                            "one c per ordered physical layout.inputs entry plus zero_ct")
        rules=rules.replace("per named output, in a flat list/tuple or a one-dimensional object array; no implicit flattening.",
                            "per physical output chunk, in ciphertext-index order, in a flat list/tuple or a one-dimensional object array.")
        rules=rules.replace("Every input uses separate C-order padded packing repeated every P=layout.input_slot_period.",
                            "Each layout.inputs entry is a separate encrypted chunk, repeated every P=layout.input_slot_period.")
    else:
        rules=rules.replace("Every logical input is encrypted separately: C-order flatten, pad to P, repeat across\n16384 slots. P=layout.input_slot_period. Never concatenate logical inputs before encryption.",
                            "Every physical layout.inputs chunk is encrypted separately and repeated across\n16384 slots. P=layout.input_slot_period. Different logical tensors never share a chunk.")
        rules=rules.replace("Return one ciphertext per named output in layout.outputs order, with logical values in\nits C-order prefix, repeated every P slots. Other slots are unobserved.",
                            "Return one ciphertext per physical output chunk in ciphertext-index order, with\nits chunk values in the C-order prefix, repeated every P slots. Other slots are unobserved.")
    return rules+"""
Chunk ABI unified-chunked-inputs-v1: logical_inputs declares tensors, inputs binds physical
DSL parameters. For each entry, flatten that logical tensor in C order, take
[offset:offset+elements], then zero-pad to P. Chunks are encrypted independently.
Each named output lists chunks with immutable ciphertext index, offset and elements.
Reassemble via output_selectors in named-output order. Cross-chunk operations must
implement logical tensor semantics; rotating a chunk alone is not a whole-tensor rotation.
Intermediate layouts are candidate choices; immutable input/output bindings cannot change.
At most four model chunks plus zero_ct and four output chunks; existing budgets apply.
"""
