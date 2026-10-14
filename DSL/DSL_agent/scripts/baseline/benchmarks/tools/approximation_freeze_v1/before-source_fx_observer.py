"""Original symbolic FX observations inside the existing source sandbox.

No DSL lowering, approximation substitution or backend operation is performed.
"""
import copy, inspect, math, operator
import numpy as np
import torch
from benchmark_graph import digest
from source_calibration_contract import (
    checked, require, FX_FORMAT, MAX_EVENTS_PER_SAMPLE, MAX_VALUES_PER_SAMPLE)

GRAPH_FORMAT = "poseidon-source-calibration-fx-graph-v1"
MAX_GRAPH_NODES = 4096

def target_name(target):
    from torch.nn import functional as F
    public = {
        F.gelu:"torch.nn.functional.gelu", F.relu:"torch.nn.functional.relu",
        F.silu:"torch.nn.functional.silu", F.softmax:"torch.nn.functional.softmax",
        F.layer_norm:"torch.nn.functional.layer_norm", F.linear:"torch.nn.functional.linear",
        F.normalize:"torch.nn.functional.normalize",
    }
    for name in ("add","sub","mul","div","neg","square","pow","mean","sum","var","matmul",
                 "mm","bmm","dot","tensordot","einsum","exp","log","sqrt","rsqrt","reciprocal",
                 "sigmoid","tanh","relu","softmax","flatten","reshape","transpose","permute",
                 "cat","stack","where","clamp"):
        if hasattr(torch,name):public[getattr(torch,name)]="torch."+name
    for name in ("add","sub","mul","truediv","neg","pow","matmul","getitem","eq","ne","gt","ge","lt","le"):
        public[getattr(operator,name)]="operator."+name
    if target in public:return public[target]
    module=getattr(target,"__module__",None);qualname=getattr(target,"__qualname__",None)
    require(type(module)is str and type(qualname)is str and "<"not in qualname,
            "calibration_fx_trace_rejected")
    return module+"."+qualname

def graph_description(graph_module):
    nodes=list(graph_module.graph.nodes)
    require(1 <= len(nodes) <= MAX_GRAPH_NODES, "calibration_observation_budget_exhausted")
    names={n.name for n in nodes}
    def wire(value):
        if isinstance(value,torch.fx.Node):
            require(value.name in names,"calibration_fx_trace_rejected")
            return {"node":value.name}
        if value is None or type(value)in(bool,int,str):return value
        if type(value)is float:
            require(math.isfinite(value),"calibration_fx_trace_rejected");return value
        if isinstance(value,torch.dtype):return {"dtype":str(value)}
        if isinstance(value,torch.device):return {"device":str(value)}
        if isinstance(value,tuple):return {"tuple":[wire(v)for v in value]}
        if isinstance(value,list):return {"list":[wire(v)for v in value]}
        if isinstance(value,dict):
            require(all(type(k)is str for k in value),"calibration_fx_trace_rejected")
            return {"dict":{k:wire(v)for k,v in sorted(value.items())}}
        if type(value)is slice:return {"slice":[wire(value.start),wire(value.stop),wire(value.step)]}
        require(False,"calibration_fx_trace_rejected")
    records=[]
    for n in nodes:
        require(n.op in ("placeholder","get_attr","call_function","call_method","call_module","output"),
                "calibration_fx_trace_rejected")
        target=target_name(n.target)if n.op=="call_function"else str(n.target)
        records.append(dict(name=n.name,op=n.op,target=target,args=wire(n.args),kwargs=wire(n.kwargs)))
    result=dict(format=GRAPH_FORMAT,nodes=records)
    result["graph_sha256"]=digest(result)
    return result

class FXObserver:
    def __init__(self,module,selection,*,tensor_elements,max_compute_nodes):
        self.selection=checked(selection)
        require(self.selection["format"]==FX_FORMAT)
        require(type(tensor_elements)is int and 1<=tensor_elements<=65536 and
                type(max_compute_nodes)is int and 1<=max_compute_nodes<=1024)
        self.tensor_elements=tensor_elements;self.module=module
        self.signature=inspect.signature(module.forward)
        require(all(p.kind not in (p.VAR_POSITIONAL,p.VAR_KEYWORD)for p in self.signature.parameters.values()),
                "calibration_fx_trace_rejected")
        try:self.traced=torch.fx.symbolic_trace(module)
        except Exception:require(False,"calibration_fx_trace_rejected")
        self.description=graph_description(self.traced)
        require(sum(n["op"]in("call_function","call_method","call_module")
                    for n in self.description["nodes"])<=max_compute_nodes,
                "calibration_observation_budget_exhausted")
        self.state={n:v.detach().clone()for n,v in self.traced.state_dict().items()}
        if self.selection["mode"]=="observe":
            require(self.selection["graph_sha256"]==self.description["graph_sha256"],
                    "calibration_fx_graph_changed")
        by_name={n["name"]:n for n in self.description["nodes"]}
        self.targets={}
        for item in self.selection["selectors"]:
            require(item["node"]in by_name,"calibration_selector_unavailable")
            actual=by_name[item["node"]]
            require(actual["op"]==item["op"]and actual["target"]==item["target"],
                    "calibration_fx_graph_changed")
            self.targets.setdefault(item["node"],[]).append(item)
        self.events=None;self.event_count=0;self.value_count=0

    def begin(self):
        self.events={item["id"]:[]for item in self.selection["selectors"]}
        self.event_count=0;self.value_count=0

    def observe(self,item,value):
        require(isinstance(value,torch.Tensor)and value.device.type=="cpu"and
                value.layout==torch.strided and value.dtype in(torch.float32,torch.float64)and
                0<value.numel()<=self.tensor_elements and value.ndim<=32,
                "calibration_projection_rejected")
        x=value.detach().clone()
        self.event_count+=1;self.value_count+=x.numel()
        require(self.event_count<=MAX_EVENTS_PER_SAMPLE and self.value_count<=MAX_VALUES_PER_SAMPLE,
                "calibration_observation_budget_exhausted")
        array=x.numpy().astype(np.float64)
        require(np.isfinite(array).all(),"calibration_nonfinite_observation")
        mean=float(array.mean());variance=float(np.mean((array-mean)**2))
        require(math.isfinite(mean)and math.isfinite(variance),"calibration_nonfinite_observation")
        lo=float(array.min());hi=float(array.max());domain=item["domain"]
        self.events[item["id"]].append(dict(
            shape=list(x.shape),dtype=str(x.dtype).removeprefix("torch."),values=int(array.size),
            minimum=lo,maximum=hi,mean=mean,variance_population=variance,
            domain_status="not_declared"if domain is None else "inside"if domain[0]<=lo and hi<=domain[1]else "outside"))

    def execute(self,*args,**kwargs):
        bound=self.signature.bind(*args,**kwargs);bound.apply_defaults()
        observer=self
        class Recorded(torch.fx.Interpreter):
            def run_node(self,node):
                selected=observer.targets.get(node.name,[])
                if selected:
                    positional,keywords=self.fetch_args_kwargs_from_env(node)
                    for item in selected:
                        if item["point"]=="input":
                            argument=item["argument"]
                            if type(argument)is int:
                                require(argument<len(positional),"calibration_selector_unavailable")
                                value=positional[argument]
                            else:
                                require(argument in keywords,"calibration_selector_unavailable")
                                value=keywords[argument]
                            observer.observe(item,value)
                value=super().run_node(node)
                for item in selected:
                    if item["point"]=="output":observer.observe(item,value)
                return value
        return Recorded(self.traced).run(*[bound.arguments[name]for name in self.signature.parameters])

    def __enter__(self):return self
    def __exit__(self,*_):
        require(graph_description(self.traced)==self.description,"calibration_fx_graph_changed")
        now=self.traced.state_dict()
        require(set(now)==set(self.state)and all(now[n].dtype==v.dtype and torch.equal(now[n],v)
                    for n,v in self.state.items()),"calibration_source_failed")
    def observations(self):
        require(self.events is not None and all(self.events[item["id"]]for item in self.selection["selectors"]),
                "calibration_selector_unavailable")
        return copy.deepcopy(self.events)

def summarize_fx(selection,samples):
    result={}
    for item in selection["selectors"]:
        events=[event for sample in samples for event in sample[item["id"]]]
        total=sum(event["values"]for event in events)
        mean=math.fsum(e["mean"]*(e["values"]/total)for e in events)
        variance=math.fsum((e["values"]/total)*(e["variance_population"]+(e["mean"]-mean)**2)for e in events)
        outside=sum(e["domain_status"]=="outside"for e in events)
        result[item["id"]]=dict(
            **{key:item[key]for key in("node","op","target","point","argument","projection")},
            samples=len(samples),calls=len(events),values=total,minimum=min(e["minimum"]for e in events),
            maximum=max(e["maximum"]for e in events),mean=mean,std_population=math.sqrt(variance),
            dtypes=sorted({e["dtype"]for e in events}),
            shapes=[list(s)for s in sorted({tuple(e["shape"])for e in events})],
            declared_domain=item["domain"],outside_domain_calls=outside,
            domain_status="not_declared"if item["domain"]is None else "outside"if outside else "inside")
    return result
