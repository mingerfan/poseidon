"""Auditable mixed joins using existing encrypted arithmetic and alias rules."""
import copy,hashlib
from pathlib import Path
import numpy as np
from benchmark_graph import require,digest,canonical
from model_resources import resolve
from pytorch_public_join_source import POLICY,VALUE_POLICY
from pytorch_public_join_values import operation,arguments,result_shape,checked_shape
from pytorch_public_value_ops import checked_array,signature
from pytorch_public_expression_flow import Ref,_decode,_refs,state_identity
from pytorch_public_mask_selection import effective_arrays
def encode(v):
    if type(v)is Ref:return dict(kind="fx_value",name=v.name)
    if type(v)in(tuple,list):return dict(kind="tuple"if type(v)is tuple else"list",items=[encode(x)for x in v])
    if type(v)is dict:return dict(kind="dict",items=[[k,encode(x)]for k,x in v.items()])
    if v is None:return dict(kind="none")
    require(type(v)in(bool,int,float),"public_join_generated_argument")
    return dict(kind={bool:"bool",int:"int",float:"float"}[type(v)],value=v)
def implementation():
    return {n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()for n in
            ("pytorch_public_join_normalization.py","pytorch_public_join_values.py","pytorch_public_join_source.py")}
def normalized_inventory(source,original,*,resource_profile):
    cap=resolve(resource_profile);capacity=cap.expanded_nodes+cap.constants+cap.inputs+cap.outputs
    require(type(source)is list and 1<=len(source)<=capacity,"public_join_inventory_capacity")
    states={n:checked_array(v,resource_profile=resource_profile)for n,v in original.items()}
    require(len(states)<=cap.constants and sum(v.size for v in states.values())<=cap.total_constant_elements,
            "public_join_original_state_capacity")
    used={row["name"]for row in source};require(len(used)==len(source),"public_join_unique_source_nodes")
    result=[];seen={};roles={};public={};entries=[];arrays={}
    def fresh(name,op,target,args,kwargs,tensor):
        require(name not in used,"public_join_generated_name_collision");used.add(name)
        row=dict(name=name,op=op,target=target,args=encode(args),kwargs=encode(kwargs),tensor=tensor)
        result.append(row);require(len(result)<=capacity,"public_join_expansion_capacity");return Ref(name)
    for item in source:
        require(type(item)is dict and set(item)=={"name","op","target","args","kwargs","tensor"}and
                type(item["name"])is str and item["name"]not in seen,"public_join_inventory_node")
        refs={n:Ref(n)for n in seen};args=_decode(item["args"],refs);kwargs=_decode(item["kwargs"],refs)
        values=_refs((args,kwargs));name=item["name"]
        roles[name]=item["op"]=="placeholder"or any(roles[v.name]for v in values)
        if item["op"]=="get_attr":
            require(item["target"]in states,"public_join_public_attribute")
            public[name]=states[item["target"]];roles[name]=False
        op=operation(item["op"],item["target"])
        if op is None:result.append(copy.deepcopy(item));seen[name]=item;continue
        op,tensors,axis=arguments(item["op"],item["target"],args,kwargs)
        require(all(type(v)is Ref and v.name in seen for v in tensors),"public_join_bound_sequence")
        require(any(roles[v.name]for v in tensors),"public_join_closed_operation_not_folded")
        metas=[seen[v.name]["tensor"]for v in tensors]
        require(all(type(v)is dict and set(v)=={"policy","shape","dtype"}for v in metas),"public_join_operand_metadata")
        dims,axis=result_shape(op,[v["shape"]for v in metas],axis,resource_profile=resource_profile)
        tensor=copy.deepcopy(item["tensor"])
        require(type(tensor)is dict and tensor["shape"]==dims and tensor["dtype"]=="float64",
                "public_join_result_metadata")
        cipher=[v for v,m in zip(tensors,metas)if roles[v.name]]
        require(all(seen[v.name]["tensor"]["dtype"]=="float64"for v in cipher),"public_join_encrypted_dtype")
        plain=[v for v in tensors if not roles[v.name]]
        generated=[];state_names=[];promotions=[];anchor=None;terms=[]
        if plain:
            anchor=cipher[0];anchor_meta=copy.deepcopy(seen[anchor.name]["tensor"])
            # Multiply each finite input element by zero before reducing; summing
            # the input first could overflow on valid finite plaintext inputs.
            zero_name=name+"_join_zero"
            zero=fresh(zero_name,"call_function","torch.mul",(anchor,0.),{},anchor_meta);generated.append(zero_name)
            if anchor_meta["shape"]:
                zero_name=name+"_join_zero_scalar"
                zero=fresh(zero_name,"call_function","torch.sum",(zero,),dict(dim=tuple(range(len(anchor_meta["shape"])))),
                    dict(anchor_meta,shape=[]));generated.append(zero_name)
            for index,v in enumerate(tensors):
                if roles[v.name]:terms.append(v);continue
                require(v.name in public,"public_join_closed_public_operand")
                data=public[v.name]
                if data.dtype.str in("<i4","<i8"):
                    require(bool(np.all(data>=-(2**53))and np.all(data<=2**53)),"public_join_integer_precision")
                remaining=cap.total_constant_elements-sum(a.size for a in states.values())
                checked_shape(list(data.shape),resource_profile=resource_profile,constant=True,remaining=remaining)
                attr="_poseidon_public_join_"+str(len(entries))+"_"+str(index)
                require(attr not in states and len(states)<cap.constants,"public_join_constant_capacity")
                converted=checked_array(np.asarray(data,dtype="<f8"),resource_profile=resource_profile)
                states[attr]=converted;arrays[attr]=converted;state_names.append(attr)
                getter=name+"_join_public_"+str(index);meta=dict(tensor,shape=list(data.shape))
                weight=fresh(getter,"get_attr",attr,(),{},meta);generated.append(getter)
                joined_name=name+"_join_cipher_"+str(index)
                term=fresh(joined_name,"call_function","torch.add",(zero,weight),{},meta);generated.append(joined_name);terms.append(term)
                promotions.append(dict(source_node=v.name,source_value=signature(data,resource_profile=resource_profile),
                    normalized_state=attr,normalized_value=signature(converted,resource_profile=resource_profile)))
        else:terms=tensors
        row=copy.deepcopy(item);row.update(args=encode((tuple(terms),)),kwargs=encode(dict(dim=axis)))
        result.append(row);seen[name]=item
        entries.append(dict(source_node=name,operation=op,axis=axis,shape=dims,source_operands=[v.name for v in tensors],
            encrypted_anchor=None if anchor is None else anchor.name,public_promotions=promotions,
            generated_nodes=generated,state_names=state_names,extra_cipher_operations=len(generated)-len(state_names)))
        require(len(result)<=capacity,"public_join_expansion_capacity")
    names={v["name"]:Ref(v["name"])for v in result}
    used_refs={v.name for row in result for v in _refs((_decode(row["args"],names),_decode(row["kwargs"],names)))}
    dropped=[row["name"]for row in result if row["op"]=="get_attr"and row["name"]not in used_refs]
    result=[row for row in result if row["name"]not in dropped]
    return result,entries,states,effective_arrays(result,states),arrays,dropped
def check(record,inputs,original,*,resource_profile,expected_source_inventory=None):
    fields={"format","input_signature","resource_profile","source_inventory","normalized_inventory","entries",
        "original_state","full_normalized_state","effective_state","dropped_get_attr_nodes","implementation",
        "constant_materialization","binding"}
    require(type(record)is dict and set(record)==fields and record.get("format")==POLICY and
        record.get("binding")==digest({k:v for k,v in record.items()if k!="binding"}),"public_join_transport_binding")
    require(record["input_signature"]==inputs and record["resource_profile"]==resource_profile and
        record["implementation"]==implementation()and record["constant_materialization"]=="zero-from-encrypted-anchor-v1",
        "public_join_source_identity")
    require(len(canonical(record))<=resolve(resource_profile).json_bytes,"public_join_record_capacity")
    if expected_source_inventory is not None:require(record["source_inventory"]==expected_source_inventory,"public_join_source_bridge")
    require(record["original_state"]==state_identity(original,resource_profile=resource_profile),"public_join_original_state_binding")
    expected,entries,full,effective,arrays,dropped=normalized_inventory(record["source_inventory"],original,resource_profile=resource_profile)
    require(record["normalized_inventory"]==expected and record["entries"]==entries and
        record["full_normalized_state"]==state_identity(full,resource_profile=resource_profile)and
        record["effective_state"]==state_identity(effective,resource_profile=resource_profile)and
        record["dropped_get_attr_nodes"]==dropped,"public_join_normalization_changed")
    return full,effective
def normalize(traced,inputs,*,resource_profile):
    import torch
    from pytorch_fused_matrix_normalization import inventory
    from pytorch_public_expression_flow import _arrays
    from pytorch_arithmetic_normalization import clone_public_state
    require(isinstance(traced,torch.fx.GraphModule)and traced.meta.get("public_expression_policy")==VALUE_POLICY,
        "public_join_explicit_policy")
    source=inventory(traced);original=_arrays(traced)
    expected,entries,full,effective,arrays,dropped=normalized_inventory(source,original,resource_profile=resource_profile)
    result=clone_public_state(traced);result.meta=copy.deepcopy(traced.meta)
    for n,v in arrays.items():result.register_buffer(n,torch.from_numpy(np.array(v,copy=True)))
    by_name={n.name:n for n in result.graph.nodes};wanted={v["name"]:v for v in expected}
    def actual(v):
        if type(v)is Ref:return by_name[v.name]
        if type(v)in(tuple,list):return type(v)(actual(x)for x in v)
        if type(v)is dict:return {k:actual(x)for k,x in v.items()}
        return v
    for entry in entries:
        node=by_name[entry["source_node"]]
        with result.graph.inserting_before(node):
            for n in entry["generated_nodes"]:
                row=wanted[n];refs={k:Ref(k)for k in by_name}
                args=_decode(row["args"],refs);kwargs=_decode(row["kwargs"],refs)
                target={"torch.mul":torch.mul,"torch.sum":torch.sum,"torch.add":torch.add}.get(row["target"],row["target"])
                fresh=result.graph.create_node(row["op"],target,actual(args),actual(kwargs),name=n)
                fresh.meta=copy.deepcopy(node.meta);fresh.meta["poseidon_static_tensor"]=copy.deepcopy(row["tensor"]);by_name[n]=fresh
        row=wanted[node.name];refs={k:Ref(k)for k in by_name}
        node.args=actual(_decode(row["args"],refs));node.kwargs=actual(_decode(row["kwargs"],refs))
    for n in dropped:
        require(not by_name[n].users,"public_join_dropped_attribute_dependency");result.graph.erase_node(by_name[n])
    result.graph.lint();result.recompile()
    require(inventory(result)==expected and inventory(traced)==source and
        state_identity(_arrays(result),resource_profile=resource_profile)==state_identity(full,resource_profile=resource_profile)and
        state_identity(_arrays(traced),resource_profile=resource_profile)==state_identity(original,resource_profile=resource_profile),
        "public_join_graph_bridge")
    record=dict(format=POLICY,input_signature=copy.deepcopy(inputs),resource_profile=resource_profile,
        source_inventory=source,normalized_inventory=expected,entries=entries,
        original_state=state_identity(original,resource_profile=resource_profile),
        full_normalized_state=state_identity(full,resource_profile=resource_profile),
        effective_state=state_identity(effective,resource_profile=resource_profile),
        dropped_get_attr_nodes=dropped,implementation=implementation(),constant_materialization="zero-from-encrypted-anchor-v1")
    record["binding"]=digest(record);check(record,inputs,original,resource_profile=resource_profile,expected_source_inventory=source)
    return result,record
