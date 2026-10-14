"""Generate only this task's trace/capture and MHA helper candidates; no apply."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
OVERLAY=HERE/"overlay"
ORIGINAL={"pytorch_static_trace.py":"bf37763c36337bb9ff7090af5ba51fad59870039c3d812739d65bd674f502080",
          "pytorch_model_capture.py":"b7d02e6b7f7c9506d1b7dbab0ac0233df513471837b10c053302373b85669d1d"}
def replace(text,old,new):
 if text.count(old)!=1:raise ValueError("Exact patch context: "+old[:80])
 return text.replace(old,new)
def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def build():
 OVERLAY.mkdir(exist_ok=True)
 snapshots={}
 for name,expected in ORIGINAL.items():
  p=BASE/name;text=p.read_text()
  if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError("Current source changed")
  snapshots[name]=text
 p=snapshots["pytorch_static_trace.py"]
 p=replace(p,'def trace(module, inputs, *, resource_profile, policy=POLICY, fused_matrix_policy=None):',
             'def trace(module, inputs, *, resource_profile, policy=POLICY, fused_matrix_policy=None, mha_source_policy=None):')
 p=replace(p,'    unknown = object()',
 '''    require(mha_source_policy is None or mha_source_policy=="poseidon-mha-ordered-source-v1" and
            policy==ALIAS_POLICY,"Explicit ordered MHA metadata policy")
    unknown = object()''')
 p=replace(p,'                child = self.root.get_submodule(target)\n                x =',
 '''                child = self.root.get_submodule(target)
                if type(child) is torch.nn.MultiheadAttention and mha_source_policy is not None:
                    from pytorch_mha_source import metadata
                    return metadata(child,args,kwargs,self.value,resource_profile=resource_profile)
                x =''')
 p=replace(p,'            split_plan=None',
 '''            mha_call=(mha_source_policy is not None and kind=="call_module" and
                      type(self.root.get_submodule(target)) is torch.nn.MultiheadAttention)
            split_plan=None''')
 p=replace(p,'                if kind == "placeholder":',
             '                if kind == "placeholder" or mha_call:')
 p=replace(p,'            self.metadata[result.node] = value',
 '''            if mha_call:
                require(type(value) is tuple and len(value)==2 and isinstance(value[0],torch.Tensor) and
                        (value[1] is None or isinstance(value[1],torch.Tensor)),"Static MHA tuple metadata")
                require(all(v is None or v.numel()<=limits.tensor_elements for v in value),"MHA tuple tensor budget")
            self.metadata[result.node] = value
            if mha_call:
                require(self.compute_count+sum(v is not None for v in value)<=limits.original_nodes,
                        "MHA tuple proxy budget")
                return tuple(None if v is None else self.create_proxy(
                    "call_function",operator.getitem,(result,i),{}) for i,v in enumerate(value))''')
 p=replace(p,'    if fused_matrix_policy is not None:\n        result.meta["fused_matrix_policy"] = fused_matrix_policy\n    return result\n',
 '''    if fused_matrix_policy is not None:
        result.meta["fused_matrix_policy"] = fused_matrix_policy
    if mha_source_policy is not None:
        result.meta["mha_source_policy"]=mha_source_policy
    return result
''')
 outputs={"pytorch_static_trace.py":p}
 p=snapshots["pytorch_model_capture.py"]
 p=replace(p,'            capture_aliases=False, gelu_source_policy="exact-gelu-source-v1"):',
             '            capture_aliases=False, gelu_source_policy="exact-gelu-source-v1", mha_source_policy=None):')
 p=replace(p,'    gelu_sources=[]',
 '''    require(mha_source_policy is None or mha_source_policy=="poseidon-mha-ordered-source-v1" and
            capture_aliases is True and preserve_output_structure is True and
            capture_policy=="signature-static-index-v1","Explicit ordered MHA capture policy")
    mha_sources=[];mha_ports={}
    gelu_sources=[]''')
 p=replace(p,'        traced=module\n',
 '''        traced=module
        if mha_source_policy is not None:
            require(traced.meta.get("mha_source_policy")==mha_source_policy,"MHA static trace policy binding")
''')
 p=replace(p,'            require(isinstance(value,torch.Tensor),"Only tensor attributes")',
 '''            require(isinstance(value,torch.Tensor),"Only tensor attributes")
            if mha_source_policy is not None:
                from pytorch_mha_source import is_mask_attribute
                if is_mask_attribute(node,traced):
                    require(value.dtype in (torch.bool,torch.float64),"Public MHA mask dtype")
                    public_values[node]=value
                    continue''')
 p=replace(p,'            m=traced.get_submodule(node.target)\n            argument=',
 '''            m=traced.get_submodule(node.target)
            if type(m) is torch.nn.MultiheadAttention and mha_source_policy is not None:
                from pytorch_mha_source import expand
                ports,source=expand(node,m,traced,cipher_ref=ref,cipher_shape=shape,
                    public_values=public_values,constant=constant,emit=emit,graph=graph,
                    public_constant_views=public_constant_views,source_origins=alias_origins,selection=selection,
                    approximations=approximations,resource_profile=resource_profile)
                mha_ports[node]=ports;mha_sources.append(source);refs.pop(node,None)
                continue
            argument=''')
 p=replace(p,'            if (capture_policy==INDEX_POLICY and target is operator.getitem and\n',
 '''            if (mha_source_policy is not None and target is operator.getitem and
                    isinstance(node.args[0],torch.fx.Node) and node.args[0] in mha_ports):
                require(len(node.args)==2 and not node.kwargs and type(node.args[1]) is int and
                        node.args[1] in (0,1),"Static MHA tuple selection")
                selected=mha_ports[node.args[0]][node.args[1]]
                require(selected is not None,"Static None must remain a literal output")
                emit(node,"dropout",[selected],dict(training=False))
                continue
            if (capture_policy==INDEX_POLICY and target is operator.getitem and
''')
 p=replace(p,'                    scalar_tensor=(lambda v:not shape(v)) if capture_policy in (POLICY,INDEX_POLICY) else None)',
 '''                    scalar_tensor=(lambda v:not shape(v)) if capture_policy in (POLICY,INDEX_POLICY) else None,
                    **({"allow_none":True} if mha_source_policy is not None else {}))''')
 p=replace(p,'    if record_state_sources:\n        require(preserve_output_structure',
 '''    if mha_source_policy is not None:
        from pytorch_mha_source import record as source_record
        binding["pytorch_capture"]["mha_source_capture"]=source_record(graph,emission_graph,mha_sources,
            source_origins=alias_origins,public_constant_views=public_constant_views,
            normalized_origins=(alias_metadata["normalization"]["mathematical_node_origins"]
                                if alias_metadata is not None else None),resource_profile=resource_profile)
    if record_state_sources:
        require(preserve_output_structure''')
 outputs["pytorch_model_capture.py"]=p
 old=BASE/"benchmarks/tools/static_mha_source_draft_v1/mha_source_plan.py"
 math_source=old.read_text()
 if sha(math_source)!="f1f845815f671091a71b9b86217c5b68644d704b736ed7a237f638dec6164f49":raise ValueError("Frozen MHA math")
 outputs["pytorch_mha_math.py"]=replace(math_source,
     '    base=Path(__file__).parents[3]\n    paths=(Path(__file__),base/"benchmark_graph.py",base/"model_decomposition.py",base/"model_resources.py")',
     '    import benchmark_graph,model_decomposition,model_resources\n    paths=(Path(__file__),*(Path(m.__file__).resolve() for m in\n            (benchmark_graph,model_decomposition,model_resources)))')
 outputs["pytorch_mha_source.py"]=(HERE/"pytorch_mha_source.py").read_text()
 for name,text in outputs.items():
  ast.parse(text,filename=name);compile(text,name,"exec")
  (OVERLAY/name).write_text(text)
 manifest=dict(format="poseidon-mha-ordered-capture-overlay-v1",
   original=ORIGINAL,candidates={n:sha(t) for n,t in outputs.items()},math_source_sha256=sha(math_source),
   production_applied=False,nix_entries=0,api=0,compiler=0,encrypted_execution=0)
 manifest["binding"]=sha(json.dumps(manifest,sort_keys=True,separators=(",",":")))
 (HERE/"capture-overlay.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 print(json.dumps(manifest))
if __name__=="__main__":build()
