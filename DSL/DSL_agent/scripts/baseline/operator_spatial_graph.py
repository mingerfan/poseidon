"""Versioned static spatial operators lowered to the existing core graph.

Public axis maps and derived convolution weights are checked before allocation.
No backend changes, model calls, generated DSL answers or runtime certification.
"""
import copy, hashlib, itertools, math, struct
from pathlib import Path
from benchmark_graph import (NAME, CAPACITY_FORMAT, canonical, digest, dimensions,
                             require, validate)
from model_decomposition import REGISTRY, REGISTRY_VERSION, Operator
from operator_arithmetic_graph import ArithmeticBuilder, FORMAT as ARITHMETIC_FORMAT, implementation_sources as arithmetic_sources
from types import MappingProxyType
from model_resources import identity as resource_identity

FORMAT="poseidon-operator-graph-v4"
VERSION="static-spatial-decomposition-v3"
RESIZE_ATTRS={"size","scale_factor","mode","align_corners","recompute_scale_factor","antialias"}
TRANSPOSE_ATTRS={"stride","padding","dilation","groups","output_padding","output_size"}

def sizes(value, rank, name, *, zero=False):
    require(type(value) is list and len(value)==rank and
            all(type(x) is int and (x>=0 if zero else x>0) for x in value),name)
    return tuple(value)

def resize_spec(shape, attrs):
    require(type(attrs) is dict and set(attrs)==RESIZE_ATTRS,"resize_attributes")
    rank=len(shape)-2
    require(rank in (1,2),"resize_rank")
    mode=attrs["mode"]
    require(mode in (("nearest","nearest-exact","linear") if rank==1 else
                     ("nearest","nearest-exact","bilinear")),"resize_mode")
    require(attrs["antialias"] is False,"resize_antialias")
    recompute=attrs["recompute_scale_factor"]
    require(recompute is None or type(recompute) is bool,"resize_recompute")
    align=attrs["align_corners"]
    require((align is None if mode.startswith("nearest") else type(align) is bool),"resize_align_corners")
    factors=attrs["scale_factor"];target=attrs["size"]
    require((target is None)!=(factors is None),"resize_size_or_factor")
    if factors is None:
        target=sizes(target,rank,"resize_size")
        require(recompute in (None,False),"resize_size_recompute")
        scales=tuple(n/m for n,m in zip(shape[2:],target))
    else:
        require(type(factors) is list and len(factors)==rank and
                all(type(x) in (float,int) and math.isfinite(x) and x>0 for x in factors),
                "resize_scale_factor")
        require(all(math.isfinite(n*f) for n,f in zip(shape[2:],factors)),"resize_output_size_budget")
        target=tuple(math.floor(n*f) for n,f in zip(shape[2:],factors))
        require(all(n>0 for n in target),"resize_empty_output")
        scales=tuple(n/m if recompute else 1/f for n,m,f in zip(shape[2:],target,factors))
    return target,scales

def f32(value):
    return struct.unpack("f",struct.pack("f",value))[0]

def row_weights(n,m,index,scale,mode,align,vectorized=False):
    if mode=="nearest":
        if vectorized:
            if n==m:return ((index,1.),)
            if m==2*n:return ((index>>1,1.),)
            position=f32(f32(index)*f32(scale))
        else:position=f32(index*scale)
        return ((min(n-1,math.floor(position)),1.),)
    if mode=="nearest-exact":
        position=(f32(f32(index+.5)*f32(scale)) if vectorized else
                  f32(max(scale*(index+.5)-.5,0.)+.5))
        return ((min(n-1,math.floor(position)),1.),)
    if n==m:return ((index,1.),)
    coordinate=(index*(n-1)/(m-1) if m>1 else 0.) if align else max(scale*(index+.5)-.5,0.)
    left=min(n-1,math.floor(coordinate));right=min(left+1,n-1)
    fraction=min(1.,max(0.,coordinate-left))
    return ((left,1-fraction),(right,fraction))

def transpose_spec(shape, weight_shape, attrs):
    require(type(attrs) is dict and set(attrs)==TRANSPOSE_ATTRS,"conv_transpose_attributes")
    rank=len(weight_shape)-2
    require(rank in (1,2) and len(shape) in (rank+1,rank+2),"conv_transpose_rank")
    batched=len(shape)==rank+2
    source=shape if batched else (1,*shape)
    stride=sizes(attrs["stride"],rank,"conv_transpose_stride")
    padding=sizes(attrs["padding"],rank,"conv_transpose_padding",zero=True)
    dilation=sizes(attrs["dilation"],rank,"conv_transpose_dilation")
    output_padding=sizes(attrs["output_padding"],rank,"conv_transpose_output_padding",zero=True)
    groups=attrs["groups"]
    require(type(groups) is int and groups>0 and source[1]%groups==0 and
            weight_shape[0]==source[1],"conv_transpose_groups_or_weight")
    minimum=tuple((n-1)*s-2*p+d*(k-1)+1 for n,s,p,d,k in
                  zip(source[2:],stride,padding,dilation,weight_shape[2:]))
    output_size=attrs["output_size"]
    ignored=None
    if output_size is not None:
        require(type(output_size) is list and len(output_size) in (rank,len(shape)) and
                all(type(x) is int and x>=0 for x in output_size),"conv_transpose_output_size")
        if len(output_size)>rank:ignored=output_size[:-rank]
        requested=tuple(output_size[-rank:])
        output_padding=tuple(o-m for o,m in zip(requested,minimum))
        require(all(0<=p<s for p,s in zip(output_padding,stride)),"conv_transpose_output_size_range")
    else:
        require(all(p<s or p<d for p,s,d in zip(output_padding,stride,dilation)),
                "conv_transpose_output_padding_range")
    output_spatial=tuple(m+p for m,p in zip(minimum,output_padding))
    require(all(n>0 for n in output_spatial),"conv_transpose_empty_output")
    return dict(rank=rank,batched=batched,source=source,stride=stride,padding=padding,
                dilation=dilation,groups=groups,output_padding=output_padding,
                output_spatial=output_spatial,ignored_output_size_leading=ignored)

def tensor(shape,fn,prefix=()):
    if not shape:return fn(prefix)
    return [tensor(shape[1:],fn,(*prefix,i)) for i in range(shape[0])]

def get(value,index):
    for i in index:value=value[i]
    return value

SPATIAL_REGISTRY=MappingProxyType({
    "resize":Operator(frozenset(RESIZE_ATTRS),"static_cpu_float64_resize","axis_public_linear",input_contract="rank 3/4, CPU Float64 contiguous semantics, no antialias"),
    "conv_transpose1d":Operator(frozenset(TRANSPOSE_ATTRS),"transposed_convolution_scatter","bounded_public_linear_or_axis_insertion_convolution",input_contract="public weight/bias, static rank 2/3 input"),
    "conv_transpose2d":Operator(frozenset(TRANSPOSE_ATTRS),"transposed_convolution_scatter","bounded_public_linear_or_axis_insertion_convolution",input_contract="public weight/bias, static rank 3/4 input"),
})

def implementation_sources():
    sources=arithmetic_sources()
    base=Path(__file__).parent
    for name in ("operator_spatial_graph.py","operator_spatial_reference.py"):
        sources[name]=hashlib.sha256((base/name).read_bytes()).hexdigest()
    return sources

class SpatialBuilder(ArithmeticBuilder):
    def __init__(self,model,*,resource_profile):
        require(type(model) is dict and model.get("format")==FORMAT,"Operator graph V4 version")
        from model_resources import resolve
        require(set(model)=={"format","id","inputs","constants","nodes","outputs"}
                and len(canonical(model))<=resolve(resource_profile).json_bytes,"Operator graph V4 fields/budget")
        prior=copy.deepcopy(model);prior["format"]=ARITHMETIC_FORMAT
        super().__init__(prior,resource_profile=resource_profile)
        self.original=copy.deepcopy(model);self.spatial_evidence=[]
    def preflight(self,shapes,matrices,derived=0):
        for s in shapes:dimensions(list(s),resource_profile=self.resource_profile)
        for m,n in matrices:
            require(m>0 and n>0 and m*n<=self.limits.constant_elements,"spatial_matrix_budget")
        existing=sum(math.prod(self.shapes[k]) for k in self.graph["constants"])
        require(existing+sum(m*n for m,n in matrices)+derived<=self.limits.total_constant_elements,
                "spatial_public_aggregate_budget")
        require(len(self.graph["constants"])+len(matrices)+int(derived>0)<=self.limits.constants,
                "spatial_constant_count_budget")
    def axis_linear(self,x,axis,weights):
        rank=len(self.shapes[x]);order=[i for i in range(rank) if i!=axis]+[axis]
        identity=list(range(rank))
        if order!=identity:x=self.one("permute",[x],dict(dims=order))
        x=self.one("linear",[x,self.constant(weights)])
        if order!=identity:
            inverse=[order.index(i) for i in identity]
            x=self.one("permute",[x],dict(dims=inverse))
        return x
    def axis_insert(self,x,axis,length,stride,left):
        # Static one-source selection followed by an exact public zero mask.
        # This is logical graph decomposition, before layout or DSL generation.
        shape=self.shapes[x];source=shape[axis]
        positions=[(i-left)//stride if i>=left and (i-left)%stride==0 and
                   0<=(i-left)//stride<source else None for i in range(length)]
        x=self.one("public_gather",[x],dict(axis=axis,indices=[j if j is not None else 0 for j in positions]))
        if any(j is None for j in positions):
            mask_shape=[1]*len(shape);mask_shape[axis]=length
            mask=tensor(mask_shape,lambda index:float(positions[index[axis]]is not None))
            x=self.one("multiply",[x,self.constant(mask)])
        return x

    def call(self,op,refs,attrs=None):
        if op not in ("resize","conv_transpose1d","conv_transpose2d"):
            return super().call(op,refs,attrs)
        require(type(refs) is list and refs and all(r in self.shapes for r in refs) and
                self.kinds[refs[0]]=="cipher","spatial_operands")
        if op=="resize":
            require(len(refs)==1,"resize_arity");x=refs[0];s=self.shapes[x]
            target,scales=resize_spec(s,attrs)
            intermediate=list(s);shapes=[];matrices=[]
            for axis,m in enumerate(target,2):
                matrices.append((m,s[axis]));intermediate[axis]=m;shapes.append(tuple(intermediate))
            self.preflight(shapes,matrices)
            # Fixed CPU Float64 contiguous semantics; dispatch affects legacy nearest indexing.
            vectorized=len(target)==2 and (sum(target)<=128 or (s[1]>3 and math.prod(s[2:])==1))
            for axis,(m,scale) in enumerate(zip(target,scales),2):
                n=self.shapes[x][axis]
                weights=[[0.]*n for _ in range(m)]
                for i in range(m):
                    for j,v in row_weights(n,m,i,scale,attrs["mode"],attrs["align_corners"],vectorized):weights[i][j]+=v
                x=self.axis_linear(x,axis,weights)
            self.spatial_evidence.append(dict(origin=self.origin,operator=op,input_shape=list(s),
                output_shape=list(self.shapes[x]),matrix_elements=sum(m*n for m,n in matrices),
                semantics="Torch 2.0.1 CPU Float64 contiguous interpolate",nearest_vectorized_path=vectorized,runtime_validated=False))
            return [x]
        rank=1 if op.endswith("1d") else 2
        require(len(refs) in (2,3) and all(self.kinds[r]=="plain" for r in refs[1:]),
                "conv_transpose_public_weights")
        x,w=refs[:2];s=self.shapes[x];ws=self.shapes[w]
        require(len(ws)==rank+2,"conv_transpose_weight_rank")
        spec=transpose_spec(s,ws,attrs);source=spec["source"]
        co=ws[1]*spec["groups"]
        require(len(refs)==2 or self.shapes[refs[2]]==(co,),"conv_transpose_bias")
        padding=spec["padding"];dilation=spec["dilation"];stride=spec["stride"]
        target=(source[0],co,*spec["output_spatial"])
        dimensions(list(target),resource_profile=self.resource_profile)
        input_features=math.prod(source[1:]);output_features=math.prod(target[1:])
        public=self.graph["constants"][w];bias_elements=output_features if len(refs)==3 else 0
        existing=sum(math.prod(self.shapes[k])for k in self.graph["constants"])
        # Use one exact public linear map when its complete storage fits the
        # existing budget. Large maps keep the independently bounded axis route.
        dense=(input_features*output_features<=self.limits.constant_elements and
               existing+input_features*output_features+bias_elements<=self.limits.total_constant_elements and
               len(self.graph["constants"])+1+int(bias_elements>0)<=self.limits.constants)
        if dense:
            self.preflight([target,(source[0],input_features),(source[0],output_features)],
                           [(output_features,input_features)],bias_elements)
            weights=[[0.]*input_features for _ in range(output_features)]
            input_strides=[math.prod(source[1:][i+1:])for i in range(len(source)-1)]
            output_strides=[math.prod(target[1:][i+1:])for i in range(len(target)-1)]
            cin_group=source[1]//spec["groups"];cout_group=ws[1]
            for cin in range(source[1]):
                group=cin//cin_group
                for location in itertools.product(*(range(n)for n in source[2:])):
                    column=sum(i*st for i,st in zip((cin,*location),input_strides))
                    for cout in range(cout_group):
                        for kernel in itertools.product(*(range(n)for n in ws[2:])):
                            dst=tuple(i*st-p+k*d for i,st,p,k,d in zip(location,stride,padding,kernel,dilation))
                            if all(0<=i<n for i,n in zip(dst,target[2:])):
                                row=sum(i*st for i,st in zip((group*cout_group+cout,*dst),output_strides))
                                weights[row][column]+=get(public,(cin,cout,*kernel))
            x=self.one("reshape",[x],dict(shape=[source[0],input_features]))
            operands=[x,self.constant(weights)]
            if len(refs)==3:
                bias=self.graph["constants"][refs[2]]
                expanded=[get(bias,(i//math.prod(target[2:]),))for i in range(output_features)]
                operands.append(self.constant(expanded))
            x=self.one("linear",operands)
            final=target if spec["batched"]else target[1:]
            x=self.one("reshape",[x],dict(shape=list(final)))
            self.spatial_evidence.append(dict(origin=self.origin,operator=op,input_shape=list(s),output_shape=list(final),
                original_weight_sha256=digest(public),transformed_weight_sha256=digest(weights),
                matrix_elements=input_features*output_features,axis_binding="exact_public_linear_map",
                ignored_output_size_leading=spec["ignored_output_size_leading"],runtime_validated=False))
            return [x]
        left=tuple(d*(k-1)-p for d,k,p in zip(dilation,ws[2:],padding))
        lifted=tuple((n-1)*st+1+2*l+p for n,st,l,p in
                     zip(source[2:],stride,left,spec["output_padding"]))
        require(all(n>0 for n in lifted),"conv_transpose_lift_shape")
        target=(source[0],co,*spec["output_spatial"])
        intermediate=list(source);shapes=[target];matrices=[]
        for axis,m in enumerate(lifted,2):
            matrices.append((m,source[axis]));intermediate[axis]=m;shapes.append(tuple(intermediate))
        out_weight_shape=(co,source[1]//spec["groups"],*ws[2:])
        self.preflight(shapes,[(m,1)for m,n in matrices],math.prod(out_weight_shape))
        if not spec["batched"]:x=self.one("reshape",[x],dict(shape=list(source)))
        for axis,(m,st,l) in enumerate(zip(lifted,stride,left),2):
            x=self.axis_insert(x,axis,m,st,l)
        public=self.graph["constants"][w]
        cin_group=source[1]//spec["groups"];cout_group=ws[1]
        def value(index):
            out,inside,*kernel=index
            group=out//cout_group
            original=(group*cin_group+inside,out%cout_group,
                      *(k-1-i for k,i in zip(ws[2:],kernel)))
            return get(public,original)
        transformed=tensor(out_weight_shape,value)
        transformed_ref=self.constant(transformed)
        operands=[x,transformed_ref]+refs[2:]
        x=self.one("conv1d" if rank==1 else "conv2d",operands,
                   dict(stride=[1]*rank,padding=[0]*rank,dilation=list(dilation),groups=spec["groups"]))
        require(self.shapes[x]==target,"conv_transpose_decomposition_shape")
        if not spec["batched"]:x=self.one("reshape",[x],dict(shape=list(target[1:])))
        self.spatial_evidence.append(dict(origin=self.origin,operator=op,input_shape=list(s),
            output_shape=list(self.shapes[x]),original_weight_sha256=digest(public),
            transformed_weight_sha256=digest(transformed),matrix_elements=0,
            public_index_elements=sum(m for m,n in matrices),axis_binding="static_gather_and_public_zero_mask",
            ignored_output_size_leading=spec["ignored_output_size_leading"],
            runtime_validated=False))
        return [x]

def decompose(model,*,resource_profile="legacy-v1"):
    b=SpatialBuilder(model,resource_profile=resource_profile);ids=set()
    for node in model["nodes"]:
        require(node["id"] not in ids,"Duplicate node id");ids.add(node["id"]);b.origin=node["id"]
        require(type(node["inputs"]) is list and all(type(r) is str and r in b.refs for r in node["inputs"]),
                "Forward/undefined operator input")
        start=len(b.graph["nodes"])
        try:values=b.call(node["op"],[b.refs[r] for r in node["inputs"]],node["attrs"])
        except ValueError as error:raise ValueError(f"node {node['id']} ({node['op']}): {error}") from error
        if node["op"] in SPATIAL_REGISTRY:
            b.spatial_evidence[-1]["core_node_ids"]=[n["id"]for n in b.graph["nodes"][start:]]
        require(type(node["outputs"]) is list and len(node["outputs"])==len(values),"Output arity")
        for name,value in zip(node["outputs"],values):b.refs[name]=value
    require(type(model["outputs"])is list,"Output declarations")
    for output in model["outputs"]:
        require(type(output)is dict and set(output)=={"name","value"}and output["value"]in b.refs,"Named output")
        b.graph["outputs"].append(dict(name=output["name"],value=b.refs[output["value"]]))
    validate(b.graph,resource_profile=resource_profile)
    if resource_profile!="legacy-v1":
        b.graph.update(format=CAPACITY_FORMAT,resource_profile=resource_profile);validate(b.graph)
    sources=implementation_sources()
    entries={**REGISTRY,**SPATIAL_REGISTRY}
    record=dict(format="poseidon-decomposition-binding-v4",registry_version=REGISTRY_VERSION+"+spatial-v3",
                original_sha256=digest(model),lowered_sha256=digest(b.graph),source_map=b.origins,
                implementation_sources=sources,implementation_sha256=digest(sources),
                required_capabilities=sorted({c for n in model["nodes"]if n["op"]in entries for c in entries[n["op"]].capabilities}),
                approximation_profiles={n["id"]:n["attrs"]for n in model["nodes"]if any(k in n["attrs"]for k in("approximation","exp","reciprocal"))},
                normalization=dict(version=VERSION,matrix_policy="static-tensor-matmul-v1",
                    matrix_entries=b.matrix_entries,arithmetic_entries=b.arithmetic_entries,spatial_entries=b.spatial_evidence,
                    scalar_carrier_shape=[1],public_constants_sha256=digest(model["constants"])),
                no_dsl_answer=True)
    if resource_profile!="legacy-v1":record.update(resource_profile=resource_identity(resource_profile),runtime_layout_validated=False)
    record["binding_sha256"]=digest(record)
    return b.graph,record

def evaluate_reference(model,inputs,ideal=False):
    from operator_spatial_reference import evaluate
    return evaluate(model,inputs,ideal=ideal)
