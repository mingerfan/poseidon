"""Passive PyTorch pre-hook observations; original forward outputs are retained."""
import copy,math
import numpy as np
import torch
from source_calibration_contract import checked,require,MAX_EVENTS_PER_SAMPLE,MAX_VALUES_PER_SAMPLE

class ModuleObserver:
    def __init__(self,module,selection,*,tensor_elements):
        self.module=module;self.selection=checked(selection)
        require(type(tensor_elements)is int and 1<=tensor_elements<=65536)
        self.tensor_elements=tensor_elements;self.handles=[];self.events=None
        self.event_count=0;self.value_count=0
        available=dict(module.named_modules())
        self.targets={}
        for item in self.selection["selectors"]:
            require(item["module"]in available,"calibration_selector_unavailable")
            target=available[item["module"]]
            if item["projection"]["kind"]=="layer_norm_variance":
                require(isinstance(target,torch.nn.LayerNorm),"calibration_projection_rejected")
            self.targets[item["id"]]=target
        self.hook_state={id(m):(m,copy.copy(m._forward_pre_hooks),copy.copy(m._forward_pre_hooks_with_kwargs))
            for m in self.targets.values()}

    def begin(self):
        self.events={item["id"]:[]for item in self.selection["selectors"]}
        self.event_count=0;self.value_count=0

    def _hook(self,item,module,args,kwargs):
        require(self.events is not None,"calibration_worker_protocol_error")
        arg=item["argument"]
        require(type(args)is tuple and type(kwargs)is dict,"calibration_projection_rejected")
        if type(arg)is int:
            require(arg<len(args),"calibration_selector_unavailable");value=args[arg]
        else:
            require(arg in kwargs,"calibration_selector_unavailable");value=kwargs[arg]
        require(isinstance(value,torch.Tensor)and value.device.type=="cpu"and
            value.layout==torch.strided and value.dtype in (torch.float32,torch.float64)and
            value.numel()>0 and value.numel()<=self.tensor_elements and value.ndim<=32,
            "calibration_projection_rejected")
        # A detached clone prevents both views and later inplace operations from
        # changing the observation. The hook always returns None.
        x=value.detach().clone()
        kind=item["projection"]["kind"]
        if kind=="rms_mean_square":
            require(x.ndim>=1,"calibration_projection_rejected")
            x=(x*x).mean(dim=-1,keepdim=True)+item["projection"]["epsilon"]
        elif kind=="layer_norm_variance":
            n=len(module.normalized_shape)
            require(n>=1 and x.ndim>=n and tuple(x.shape[-n:])==tuple(module.normalized_shape),
                    "calibration_projection_rejected")
            x=x.var(dim=tuple(range(x.ndim-n,x.ndim)),unbiased=False,keepdim=True)+module.eps
        self.event_count+=1;self.value_count+=x.numel()
        require(self.event_count<=MAX_EVENTS_PER_SAMPLE and self.value_count<=MAX_VALUES_PER_SAMPLE,
                "calibration_observation_budget_exhausted")
        array=x.numpy().astype(np.float64)
        require(np.isfinite(array).all(),"calibration_nonfinite_observation")
        mean=float(array.mean());variance=float(np.mean((array-mean)**2))
        require(math.isfinite(mean)and math.isfinite(variance),"calibration_nonfinite_observation")
        minimum=float(array.min());maximum=float(array.max());domain=item["domain"]
        self.events[item["id"]].append(dict(shape=list(x.shape),dtype=str(x.dtype).removeprefix("torch."),
            values=int(array.size),minimum=minimum,maximum=maximum,mean=mean,
            variance_population=variance,domain_status="not_declared"if domain is None else
            "inside"if domain[0]<=minimum and maximum<=domain[1]else "outside"))
        return None

    def __enter__(self):
        for item in self.selection["selectors"]:
            def hook(module,args,kwargs,item=item):return self._hook(item,module,args,kwargs)
            self.handles.append(self.targets[item["id"]].register_forward_pre_hook(hook,with_kwargs=True))
        return self
    def __exit__(self,*_):
        for handle in reversed(self.handles):handle.remove()
        self.handles=[]
        require(all(m._forward_pre_hooks==before and m._forward_pre_hooks_with_kwargs==flags
                    for m,before,flags in self.hook_state.values()),"calibration_observer_changed_forward")
    def observations(self):
        require(all(self.events[item["id"]]for item in self.selection["selectors"]),
                "calibration_selector_unavailable")
        return copy.deepcopy(self.events)

def summarize(selection,samples):
    selection=checked(selection)
    require(type(samples)is list and 1<=len(samples)<=16,"calibration_worker_protocol_error")
    result={}
    for item in selection["selectors"]:
        events=[event for sample in samples for event in sample[item["id"]]]
        total=sum(event["values"]for event in events)
        mean=sum(e["values"]*e["mean"]for e in events)/total
        variance=sum(e["values"]*(e["variance_population"]+(e["mean"]-mean)**2)for e in events)/total
        result[item["id"]]=dict(module=item["module"],argument=item["argument"],projection=item["projection"],
            samples=len(samples),calls=len(events),values=total,
            minimum=min(e["minimum"]for e in events),maximum=max(e["maximum"]for e in events),
            mean=mean,std_population=math.sqrt(variance),
            dtypes=sorted({e["dtype"]for e in events}),shapes=sorted({tuple(e["shape"])for e in events}),
            declared_domain=item["domain"],outside_domain_calls=sum(e["domain_status"]=="outside"for e in events),
            domain_status="not_declared"if item["domain"]is None else
             "outside"if any(e["domain_status"]=="outside"for e in events)else "inside")
        result[item["id"]]["shapes"]=[list(shape)for shape in result[item["id"]]["shapes"]]
    return result
