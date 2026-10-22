"""Public boolean selection lowering to registered encrypted arithmetic."""
import copy,hashlib
from pathlib import Path
import numpy as np
from benchmark_graph import require,digest,canonical
from model_resources import resolve
from pytorch_public_mask_source import POLICY,SOURCE_FORMAT
from pytorch_public_mask_values import operation,arguments,shape,result_shape,scalar,fill
from pytorch_public_value_ops import checked_array,signature
from pytorch_public_expression_flow import Ref,_decode,_refs,state_identity
def selected(spec):
    return type(spec)is dict and spec.get("format")==SOURCE_FORMAT
def target_name(kind,target):
    if kind=="call_method":return target if type(target)is str else None
    if kind=="call_function":
        import torch
        return "torch.where"if target is torch.where else None
    return None
def metadata(kind,target,args,kwargs,*,resolve_value,public_value,resource_profile):
    stable=target_name(kind,target)
    if operation(kind,stable)is None:return None
    import torch
    from torch.fx import Proxy
    op,condition,left,right=arguments(kind,stable,args,kwargs)
    public=public_value(condition)
    require(public is not None and public.dtype.str=="|b1","public_mask_declared_boolean_condition")
    def branch(value):
        if isinstance(value,Proxy):
            meta=resolve_value(value);raw=public_value(value)
            require(isinstance(meta,torch.Tensor)and
                    meta.dtype in(torch.float32,torch.float64,torch.int32,torch.int64,torch.bool),"public_mask_value_dtype")
            if raw is None:require(meta.dtype is torch.float64,"public_mask_encrypted_float64_input")
            return list(meta.shape),raw is None
        scalar(value);return [],False
    adims,acipher=branch(left)
    if op=="where":
        bdims,bcipher=branch(right)
        require(acipher or bcipher,"public_mask_encrypted_branch")
    else:
        require(acipher,"public_mask_encrypted_branch")
        value=public_value(right)
        if value is None:fill(scalar(right),"<f8")
        else:require(value.ndim==0,"public_mask_scalar_fill");fill(value.item(),"<f8")
        bdims=[]
    dims=result_shape(list(public.shape),adims,bdims,resource_profile=resource_profile,inplace=op=="masked_fill_")
    return torch.empty(dims,device="meta",dtype=torch.float64)
def encode(value):
    if type(value)is Ref:return dict(kind="fx_value",name=value.name)
    if type(value)is tuple:return dict(kind="tuple",items=[encode(v)for v in value])
    if type(value)is dict:return dict(kind="dict",items=[[k,encode(v)]for k,v in value.items()])
    require(type(value)in(bool,int,float),"public_mask_generated_argument")
    return dict(kind={bool:"bool",int:"int",float:"float"}[type(value)],value=value)
def implementation():
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()for name in
            ("pytorch_public_mask_selection.py","pytorch_public_mask_values.py","pytorch_public_mask_source.py")}
def effective_arrays(source,states):
    out={v["target"]:states[v["target"]]for v in source if v["op"]=="get_attr"}
    modules=[v["target"]+"."for v in source if v["op"]=="call_module"]
    out.update({name:value for name,value in states.items()if any(name.startswith(prefix)for prefix in modules)})
    return out
def normalized_inventory(source,original,*,resource_profile):
    cap=resolve(resource_profile);capacity=cap.expanded_nodes+cap.constants+cap.inputs+cap.outputs
    require(type(source)is list and 1<=len(source)<=capacity,"public_mask_inventory_capacity")
    require(type(original)is dict and all(type(n)is str and n for n in original),"public_mask_original_state")
    states={n:checked_array(v,resource_profile=resource_profile)for n,v in original.items()}
    require(len(states)<=cap.constants and sum(v.size for v in states.values())<=cap.total_constant_elements,
            "public_mask_original_state_budget")
    used={v["name"]for v in source};seen={};roles={};public={};result=[];entries=[];arrays={}
    require(len(used)==len(source),"public_mask_unique_source_nodes")
    def add_public(value,name,node_name,tensor):
        require(name not in states and node_name not in used,"public_mask_generated_name_collision")
        require(len(states)<cap.constants,"public_mask_constant_count_budget")
        remaining=cap.total_constant_elements-sum(v.size for v in states.values())
        shape(list(value.shape),resource_profile=resource_profile,constant=True,remaining=remaining)
        value=checked_array(np.asarray(value,dtype="<f8"),resource_profile=resource_profile)
        states[name]=value;arrays[name]=value;used.add(node_name)
        row=dict(name=node_name,op="get_attr",target=name,args=encode(()),kwargs=encode({}),
                 tensor=dict(tensor,shape=list(value.shape),dtype="float64"))
        result.append(row);return Ref(node_name)
    def add_compute(target,args,name,tensor,kind="call_function"):
        require(name not in used,"public_mask_generated_name_collision");used.add(name)
        result.append(dict(name=name,op=kind,target=target,args=encode(args),kwargs=encode({}),tensor=tensor))
        return Ref(name)
    for item in source:
        require(type(item)is dict and set(item)=={"name","op","target","args","kwargs","tensor"}and
                type(item["name"])is str and item["name"]not in seen,"public_mask_inventory_node")
        names={n:Ref(n)for n in seen};args=_decode(item["args"],names);kwargs=_decode(item["kwargs"],names)
        require(type(args)is tuple and type(kwargs)is dict,"public_mask_inventory_arguments")
        refs=_refs((args,kwargs));name=item["name"]
        roles[name]=item["op"]=="placeholder"or any(roles[v.name]for v in refs)
        if item["op"]=="get_attr":
            require(item["target"]in states,"public_mask_public_attribute")
            public[name]=states[item["target"]];roles[name]=False
        op=operation(item["op"],item["target"])
        if op is None:
            result.append(copy.deepcopy(item));seen[name]=item;continue
        op,condition,left,right=arguments(item["op"],item["target"],args,kwargs)
        require(type(condition)is Ref and condition.name in public and public[condition.name].dtype.str=="|b1",
                "public_mask_declared_boolean_condition")
        mask=public[condition.name]
        def branch(value):
            if type(value)is Ref:
                require(value.name in seen,"public_mask_operand_binding")
                tensor=seen[value.name]["tensor"]
                require(type(tensor)is dict and set(tensor)=={"policy","shape","dtype"},
                        "public_mask_operand_metadata")
                if roles[value.name]:
                    require(tensor["dtype"]=="float64","public_mask_encrypted_float64_input")
                    return dict(kind="cipher",name=value.name,shape=shape(tensor["shape"],resource_profile=resource_profile))
                require(value.name in public,"public_mask_closed_public_branch")
                data=public[value.name]
                return dict(kind="public",name=value.name,shape=list(data.shape),data=data)
            value=scalar(value);return dict(kind="scalar",value=value,shape=[],data=np.asarray(value))
        a=branch(left);bb=branch(right)
        require(a["kind"]=="cipher"or op=="where"and bb["kind"]=="cipher","public_mask_encrypted_branch")
        if op!="where":
            require(a["kind"]=="cipher"and not bb["shape"]and bb["kind"]!="cipher","public_mask_scalar_fill")
            fill(bb["data"].item(),"<f8")
        dims=result_shape(list(mask.shape),a["shape"],bb["shape"],resource_profile=resource_profile,inplace=op=="masked_fill_")
        tensor=copy.deepcopy(item["tensor"])
        require(tensor==dict(policy="signature-static-ordered-alias-staging-v1",shape=dims,dtype="float64"),
                "public_mask_result_metadata")
        generated=[];state_names=[];proof_operands=[]
        for value in(a,bb):
            proof_operands.append({k:v for k,v in value.items()if k!="data"})
        choices=((a,mask),(bb,np.logical_not(mask)))if op=="where"else((a,np.logical_not(mask)),(bb,mask))
        terms=[]
        for index,(value,chosen)in enumerate(choices):
            attr="_poseidon_public_mask_"+str(len(entries))+"_"+str(index)
            getter=name+"_mask_public_"+str(index)
            if value["kind"]=="cipher":
                factor=chosen.astype("<f8")
                factor_ref=add_public(factor,attr,getter,tensor)
                step_name=name+"_mask_mul_"+str(index)
                step_kind="call_method"if op=="masked_fill_"else"call_function"
                step_target="mul_"if op=="masked_fill_"else"torch.mul"
                step_dims=result_shape(list(chosen.shape),value["shape"],[],resource_profile=resource_profile,inplace=op=="masked_fill_")
                term=add_compute(step_target,(Ref(value["name"]),factor_ref),step_name,
                                 dict(tensor,shape=step_dims),kind=step_kind)
                generated.extend((getter,step_name))
            else:
                chosen_dims=result_shape(list(chosen.shape),value["shape"],[],resource_profile=resource_profile,constant=True)
                remaining=cap.total_constant_elements-sum(v.size for v in states.values())
                shape(chosen_dims,resource_profile=resource_profile,constant=True,remaining=remaining)
                with np.errstate(over="ignore",invalid="ignore"):
                    contribution=chosen.astype("<f8")*np.asarray(value["data"],dtype="<f8")
                term=add_public(contribution,attr,getter,tensor);generated.append(getter)
            state_names.append(attr);terms.append(term)
        row=copy.deepcopy(item);row.update(op="call_method"if op=="masked_fill_"else"call_function",
            target="add_"if op=="masked_fill_"else"torch.add",args=encode(tuple(terms)),kwargs=encode({}))
        result.append(row)
        entries.append(dict(source_node=name,source_kind=item["op"],source_target=item["target"],
            condition=condition.name,condition_value=signature(mask,resource_profile=resource_profile),
            operands=proof_operands,result_shape=dims,inplace=op=="masked_fill_",
            generated_nodes=generated,state_names=state_names,encrypted_condition=False))
        seen[name]=item
        require(len(result)<=capacity,"public_mask_expansion_budget")
    names={v["name"]:Ref(v["name"])for v in result}
    used_refs={v.name for row in result for v in _refs((_decode(row["args"],names),_decode(row["kwargs"],names)))}
    dropped=[row["name"]for row in result if row["op"]=="get_attr"and row["name"]not in used_refs]
    result=[row for row in result if row["name"]not in dropped]
    return result,entries,states,effective_arrays(result,states),arrays,dropped
def check(record,inputs,original,*,resource_profile,expected_source_inventory=None):
    fields={"format","input_signature","resource_profile","source_inventory","normalized_inventory","entries",
        "original_state","full_normalized_state","effective_state","dropped_get_attr_nodes","encrypted_conditions","implementation","binding"}
    require(type(record)is dict and set(record)==fields and record.get("format")==POLICY and
            record.get("binding")==digest({k:v for k,v in record.items()if k!="binding"}),"public_mask_transport_binding")
    require(record["input_signature"]==inputs and record["resource_profile"]==resource_profile and
            record["implementation"]==implementation(),"public_mask_source_identity")
    require(len(canonical(record))<=resolve(resource_profile).json_bytes,"public_mask_record_budget")
    if expected_source_inventory is not None:
        require(record["source_inventory"]==expected_source_inventory,"public_mask_source_bridge")
    require(record["original_state"]==state_identity(original,resource_profile=resource_profile),"public_mask_original_state_binding")
    expected,entries,full,effective,arrays,dropped=normalized_inventory(record["source_inventory"],original,resource_profile=resource_profile)
    require(record["normalized_inventory"]==expected and record["entries"]==entries and
            record["full_normalized_state"]==state_identity(full,resource_profile=resource_profile)and
            record["effective_state"]==state_identity(effective,resource_profile=resource_profile)and
            record["dropped_get_attr_nodes"]==dropped and record["encrypted_conditions"]is False,
            "public_mask_normalization_changed")
    return full,effective
def normalize(traced,inputs,*,resource_profile):
    import torch
    from pytorch_fused_matrix_normalization import inventory
    from pytorch_public_expression_flow import _arrays
    from pytorch_arithmetic_normalization import clone_public_state
    require(isinstance(traced,torch.fx.GraphModule)and traced.meta.get("public_mask_source_policy")==POLICY,
            "public_mask_explicit_policy")
    source=inventory(traced);original=_arrays(traced)
    expected,entries,full,effective,arrays,dropped=normalized_inventory(source,original,resource_profile=resource_profile)
    result=clone_public_state(traced);result.meta=copy.deepcopy(traced.meta)
    for name,value in arrays.items():result.register_buffer(name,torch.from_numpy(np.array(value,copy=True)))
    by_name={n.name:n for n in result.graph.nodes};wanted={v["name"]:v for v in expected}
    for entry in entries:
        node=by_name[entry["source_node"]]
        with result.graph.inserting_before(node):
            for name in entry["generated_nodes"]:
                row=wanted[name];refs={n:Ref(n)for n in by_name}
                args=_decode(row["args"],refs);kwargs=_decode(row["kwargs"],refs)
                def actual(value):
                    if type(value)is Ref:return by_name[value.name]
                    if type(value)is tuple:return tuple(actual(v)for v in value)
                    if type(value)is dict:return {k:actual(v)for k,v in value.items()}
                    return value
                target={"torch.mul":torch.mul}[row["target"]]if row["op"]=="call_function"else row["target"]
                fresh=result.graph.create_node(row["op"],target,actual(args),actual(kwargs),name=name)
                fresh.meta=copy.deepcopy(node.meta);fresh.meta["poseidon_static_tensor"]=copy.deepcopy(row["tensor"])
                by_name[name]=fresh
        row=wanted[node.name];args=_decode(row["args"],{n:Ref(n)for n in by_name})
        node.op=row["op"];node.target="add_"if row["op"]=="call_method"else torch.add
        node.args=tuple(by_name[v.name]for v in args);node.kwargs={}
    for name in dropped:
        require(not by_name[name].users,"public_mask_dropped_attribute_dependency");result.graph.erase_node(by_name[name])
    result.graph.lint();result.recompile()
    require(inventory(result)==expected and inventory(traced)==source and
            state_identity(_arrays(result),resource_profile=resource_profile)==state_identity(full,resource_profile=resource_profile)and
            state_identity(_arrays(traced),resource_profile=resource_profile)==state_identity(original,resource_profile=resource_profile),
            "public_mask_graph_bridge")
    record=dict(format=POLICY,input_signature=copy.deepcopy(inputs),resource_profile=resource_profile,
        source_inventory=source,normalized_inventory=expected,entries=entries,
        original_state=state_identity(original,resource_profile=resource_profile),
        full_normalized_state=state_identity(full,resource_profile=resource_profile),
        effective_state=state_identity(effective,resource_profile=resource_profile),
        dropped_get_attr_nodes=dropped,encrypted_conditions=False,implementation=implementation())
    record["binding"]=digest(record)
    check(record,inputs,original,resource_profile=resource_profile,expected_source_inventory=source)
    return result,record
