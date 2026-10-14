"""RuntimePlan integrity, fixed capacity and instruction-state checks for Poseidon v1."""
import hashlib,json,math,struct,copy
from pathlib import Path
PRIMES=(1152921504581419009,1152921504581877761,1152921504583647233,
1152921504585547777,1152921504586530817,1152921504589938689,
1152921504592429057,1152921504592822273,1152921504593412097,
1152921504595640321,1152921504595968001,1152921504597016577,
1152921504598720513,1152921504606584833)
CONTEXT="poseidon-agent-n15-q13-p1-tc128-v1"
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(raw):return "sha256:"+hashlib.sha256(raw).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def read(path,limit=16*1024**2):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=limit,"Artifact file boundary")
    return path.read_bytes()
def json_read(raw):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,"Duplicate JSON key");out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError("Nonfinite JSON")))
def payload_values(plan,operation,bundle):
    payload=operation["payload"]
    if payload["kind"]=="inline":
        values=payload["values"]
    else:
        require(payload["kind"]=="bundle","Unsupported plaintext payload")
        ref=plan["plaintext_bundle"];raw=read(bundle/"manifest.json")
        require(sha(raw)==ref["manifest_sha256"],"Bundle manifest hash")
        manifest=json_read(raw)
        require(manifest["bundle_format_version"]==1 and manifest["bundle_id"]==ref["id"] and manifest["version"]==ref["version"],"Bundle identity")
        content=payload["content"]
        require(type(content)is str and len(content)==71 and content.startswith("sha256:") and all(c in "0123456789abcdef" for c in content[7:]),"Bundle content name")
        matches=[b for b in manifest["blobs"] if b["content"]==content];require(len(matches)==1,"Bundle entry count")
        raw=read(bundle/"data"/(content[7:]+".bin"))
        require(len(raw)==matches[0]["byte_length"] and len(raw)%8==0 and sha(raw)==content,"Bundle bytes hash")
        values=list(struct.unpack("<"+"d"*(len(raw)//8),raw))
    require(type(values)is list and 1<=len(values)<=16384 and all(type(x)in(int,float) and math.isfinite(x) for x in values),"Plaintext vector bounds")
    return values

def inspect(plan_path,spec_path,request,*,bundle=None):
    raw=read(plan_path);plan=json_read(raw);spec_raw=read(spec_path);spec=json_read(spec_raw)
    from compiler_configuration import request_configuration
    config=request_configuration(request)
    require(config is not None and config["backend"]=="poseidon_cpu" and
            config["operator_spec_sha256"]==hashlib.sha256(spec_raw).hexdigest(),"Request backend/spec binding")
    require(plan["format_version"]==1 and not plan["finalization"],"Plan version/finalization")
    require(plan["target"]==dict(capability_version=1,device_counts=[0],operator_spec=dict(id="poseidon-agent-cpu-v1",source_sha256=sha(spec_raw),version=1),target_id="poseidon-ckks-cpu",world_size=1),"Plan target identity")
    require(spec["spec_id"]=="poseidon-agent-cpu-v1" and spec["context"]==dict(context_id=CONTEXT,poly_degree=32768,rns_moduli_log2=[60]*13,max_modulus_log2=60,default_scale_log2=45),"OperatorSpec parameter identity")
    require(spec["levels"]==dict(lower_bound=0,upper_bound=12) and not spec["operators"]["boot"]["supported"] and not spec["boot_profiles"],"OperatorSpec level/bootstrap policy")
    from candidate_contract import request_input_names,request_rotations
    require(len(plan["external_inputs"])==len(request_input_names(request)) and len(plan["final_outputs"])==request["layout"]["output_ciphertexts"],"Plan I/O count")
    require(len(plan["values"])<=8192 and len(plan["initialization"])+len(plan["execution"])<=4096,"Plan resource bound")
    bundle_encode=any(op.get("kind")=="encode" and op.get("payload",{}).get("kind")=="bundle"
                      for op in [*plan["initialization"],*plan["execution"]])
    require(("plaintext_bundle" in plan)==bundle_encode,
            "plaintext_bundle must be present exactly when bundle Encode is used")
    values={};defined=set();states={}
    for v in plan["values"]:
        id=v["id"];require(id not in values,"Duplicate value")
        require(v["context"]==CONTEXT and v["place"]==dict(kind="host",rank=0) and v["ntt"] is True,"Value context/place")
        require(type(v["level"])is int and 0<=v["level"]<=12 and type(v["scale_log2"])is int and 0<=v["scale_log2"]<math.prod(PRIMES[:v["level"]+1]).bit_length(),"Value scale/modulus capacity")
        require(v["kind"]in("plaintext","ciphertext") and type(v["components"])is int and 1<=v["components"]<=3,"Value type")
        values[id]=v
    for id in plan["external_inputs"]:
        v=values[id];require(id not in defined and (v["kind"],v["level"],v["scale_log2"],v["components"])==("ciphertext",12,config["waterline"],2),"Input declaration")
        defined.add(id);states[id]=v["scale_log2"]
    rotations=set(request_rotations(request));ordinals=set();drops=[]
    for op in [*plan["initialization"],*plan["execution"]]:
        ordinal=op["ordinal"];label="instruction "+str(ordinal)+": "
        require(ordinal not in ordinals,"Duplicate ordinal");ordinals.add(ordinal)
        dst=op["output"];require(dst in values and dst not in defined,label+"SSA output")
        v=values[dst]
        if op["kind"]=="encode":
            require(v["kind"]=="plaintext" and v["components"]==1,label+"encode type")
            slots=payload_values(plan,op,Path(bundle or "."))
            magnitude=max(abs(x) for x in slots)
            capacity=math.prod(PRIMES[:v["level"]+1]).bit_length()
            require(magnitude==0 or math.ceil(math.log2(magnitude))+v["scale_log2"]+1<capacity,
                    label+"plaintext encoding capacity")
            state=v["scale_log2"]
        else:
            require(op["kind"]=="compute" and op["place"]==dict(kind="host",rank=0),label+"unsupported placement/instruction")
            src=op["inputs"];require(src and all(i in defined for i in src),label+"undefined input")
            a=values[src[0]];state=states[src[0]];kind=op["op"];attrs=op.get("attrs",{})
            require(a["kind"]=="ciphertext" and v["kind"]=="ciphertext",label+"ciphertext type")
            require(kind in spec["operators"] and spec["operators"][kind]["supported"],label+"unsupported capability: "+kind)
            level=a["level"];components=a["components"]
            if kind in ("add_cc","add_cp","sub_cc","sub_cp","mul_cc","mul_cp"):
                require(len(src)==2,label+"binary arity");b=values[src[1]]
                require(b["kind"]==("ciphertext" if kind.endswith("cc") else "plaintext") and b["level"]==level,label+"operand type/level")
                if kind.startswith("mul"):
                    state+=states[src[1]];components=a["components"]+b["components"]-1
                else:
                    require(state==states[src[1]],label+"addition/subtraction scale mismatch")
                    components=max(a["components"],b["components"])
            else:
                require(len(src)==1,label+"unary arity")
                if kind=="mod_switch":
                    level=attrs["target_level"];require(0<=level<=a["level"],label+"modswitch target")
                elif kind=="rescale":
                    level=attrs["target_level"];require(type(level)is int and level==a["level"]-1,label+"single rescale/chain exhaustion")
                    actual=state-math.log2(PRIMES[a["level"]]);target=attrs["target_scale_log2"]
                    # Runtime resets scale to 2^target. Only the exact pinned-prime discrepancy is allowed.
                    expected=state-60
                    require(target==expected,label+"rescale scale override would alter interpretation")
                    drops.append(dict(ordinal=ordinal,prime=PRIMES[a["level"]],actual_log2=actual,declared_log2=target))
                    state=target
                elif kind=="rotate":require(attrs["steps"] in rotations,label+"unrequested rotation")
                elif kind=="relinearize":require(components==3,label+"relinearize input");components=2
                else:require(kind=="negate",label+"unsupported capability: "+kind)
            require((v["level"],v["scale_log2"],v["components"])==(level,state,components),label+"output declaration differs from propagation")
        states[dst]=state;defined.add(dst)
    require(all(x in defined and values[x]["kind"]=="ciphertext" for x in plan["final_outputs"]),"Final outputs")
    return dict(format="poseidon-artifact-gate-v1",plan_sha256=sha(raw),spec_sha256=sha(spec_raw),
        instruction_count=len(ordinals),input_count=len(plan["external_inputs"]),output_count=len(plan["final_outputs"]),
        rescale_rounding=drops,level_convention="zero-based data modulus index",q=list(PRIMES[:13]),p=list(PRIMES[13:]),security="tc128")

def bind_periodic_constants(plan_path,bundle_path,output,period,*,capacity_layout=None):
    """Materialize the existing Hecate periodic layout through standard bundle payloads."""
    from packed_input_abi import rotations
    rotations(period,capacity_layout=capacity_layout)
    raw=read(plan_path);plan=json_read(raw);output=Path(output);output.mkdir()
    dest=output/"bundle";(dest/"data").mkdir(parents=True)
    entries={};mapping=[]
    for op in [*plan["initialization"],*plan["execution"]]:
        if op["kind"]!="encode":continue
        slots=payload_values(plan,op,Path(bundle_path))
        require((len(slots)<=period and period%len(slots)==0) or (len(slots)==16384 and all(x==slots[i%period] for i,x in enumerate(slots))),"Constant does not fit periodic layout")
        full=slots*(16384//len(slots));data=struct.pack("<"+"d"*16384,*full);content=sha(data)
        (dest/"data"/(content[7:]+".bin")).write_bytes(data)
        entries[content]=dict(content=content,byte_length=len(data));mapping.append(dict(output=op["output"],source_elements=len(slots),content=content))
        op["payload"]=dict(kind="bundle",content=content)
    manifest=dict(bundle_format_version=1,bundle_id="agent-periodic-v1",version=1,blobs=list(entries.values()))
    manifest_raw=canonical(manifest);(dest/"manifest.json").write_bytes(manifest_raw)
    if entries:
        plan["plaintext_bundle"]=dict(id="agent-periodic-v1",version=1,manifest_sha256=sha(manifest_raw))
    else:
        # The RuntimePlan protocol forbids a bundle reference without a bundle Encode.
        # Keep the empty auxiliary manifest for package/path compatibility.
        plan.pop("plaintext_bundle",None)
    (output/"plan.json").write_bytes(canonical(plan))
    record=dict(format="poseidon-periodic-binding-v1",source_plan_sha256=sha(raw),slot_period=period,slots=16384,mapping=mapping,plan_sha256=sha(canonical(plan)))
    if period>256:
        record.update(format="poseidon-periodic-binding-v2",execution_abi=capacity_layout["execution_abi"],
                      layout_sha256=sha(canonical(capacity_layout)))
    (output/"layout-binding.json").write_bytes(canonical(record))
    return record
