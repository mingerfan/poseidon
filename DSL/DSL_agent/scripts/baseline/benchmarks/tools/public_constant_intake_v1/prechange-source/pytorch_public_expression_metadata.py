"""Closed declared public values for the explicitly selected signature tracer."""
from dataclasses import dataclass
from benchmark_graph import require
from model_resources import resolve
from pytorch_public_value_ops import POLICY,ARITHMETIC_POLICY,TRANSFORM_POLICY,ARITHMETIC_POLICIES,checked_policy,checked_array,materialize,plan
from pytorch_public_tensor_transforms import registered as transform_registered
from pytorch_public_arithmetic_values import METHODS as ARITHMETIC_METHODS,plan as arithmetic_plan
from pytorch_public_expression_flow import _arrays,state_identity,METHODS

@dataclass(frozen=True)
class Selection:
    logical:object
    physical:object
    proof:object

class PublicMetadata:
    def __init__(self,module,*,resource_profile,policy=POLICY):
        import torch,operator,builtins
        self.module=module;self.profile=resource_profile;self.limits=resolve(resource_profile);self.policy=checked_policy(policy)
        raw=_arrays(module)
        require(len(raw)<=self.limits.constants,"public_expression_state_capacity")
        self.original={n:checked_array(v,resource_profile=resource_profile)for n,v in raw.items()}
        require(sum(v.size for v in self.original.values())<=self.limits.total_constant_elements,
                "public_expression_state_capacity")
        self.identity=state_identity(self.original,resource_profile=resource_profile)
        self.values={};self.derived_count=0;self.derived_elements=0
        self.functions={torch.reshape:"torch.reshape",torch.permute:"torch.permute",
                        torch.transpose:"torch.transpose",torch.t:"torch.t",
                        torch.unsqueeze:"torch.unsqueeze",torch.squeeze:"torch.squeeze",
                        torch.flatten:"torch.flatten",operator.getitem:"operator.getitem",
                        builtins.getattr:"builtins.getattr"}
        if self.policy==TRANSFORM_POLICY:
            self.functions.update({getattr(torch,name):"torch."+name for name in
                                   ("tile","repeat_interleave","flip","roll")})
        self.arithmetic_functions={operator.add:"operator.add",operator.sub:"operator.sub",
            operator.mul:"operator.mul",operator.truediv:"operator.truediv",operator.neg:"operator.neg",
            torch.add:"torch.add",torch.sub:"torch.sub",torch.subtract:"torch.subtract",
            torch.mul:"torch.mul",torch.multiply:"torch.multiply",torch.div:"torch.div",
            torch.divide:"torch.divide",torch.true_divide:"torch.true_divide",
            torch.neg:"torch.neg",torch.negative:"torch.negative"}
    def references(self,value):
        from torch.fx import Proxy
        if isinstance(value,Proxy):return [value]
        if type(value)in(tuple,list):return [p for v in value for p in self.references(v)]
        if type(value)is dict:return [p for v in value.values()for p in self.references(v)]
        return []
    def static(self,value):
        import torch
        from torch.fx import Proxy
        if isinstance(value,Proxy):return value.node.name
        if type(value)is torch.Size:return tuple(int(v)for v in value)
        if value is torch.float32:return "torch.float32"
        if value is torch.float64:return "torch.float64"
        if type(value)in(tuple,list):return type(value)(self.static(v)for v in value)
        if type(value)is dict:return {k:self.static(v)for k,v in value.items()}
        return value
    def select(self,kind,target,args,kwargs):
        from torch.fx import Proxy
        if kind=="get_attr":
            require(type(target)is str and target in self.original,"public_expression_attribute_not_declared")
            value=self.original[target]
            return Selection(value,materialize(value,resource_profile=self.profile),None)
        if kind not in("call_function","call_method","call_module"):return None
        references=self.references((args,kwargs))
        first=args[0]if args else kwargs.get("input")
        stable=target if kind=="call_method"and type(target)is str and (target in METHODS or
            self.policy==TRANSFORM_POLICY and transform_registered(kind,target)) else self.functions.get(target)if kind=="call_function"else None
        first_public=isinstance(first,Proxy)and first.node in self.values
        if stable is not None and first_public:
            require(all(p.node in self.values for p in references),"public_expression_dynamic_metadata")
        if not references or any(p.node not in self.values for p in references):return None
        arithmetic_target=(target if kind=="call_method"and type(target)is str and target in ARITHMETIC_METHODS
            else self.arithmetic_functions.get(target)if kind=="call_function"else None)
        arithmetic=self.policy in ARITHMETIC_POLICIES and arithmetic_target is not None
        require(stable is not None or arithmetic,"public_expression_operation_not_registered")
        if arithmetic:
            def public_operand(value):
                return (value.node.name,self.values[value.node].logical)if isinstance(value,Proxy)and value.node in self.values else None
            logical,physical,proof=arithmetic_plan(kind,arithmetic_target,args,kwargs,
                resolve_value=public_operand,resource_profile=self.profile)
        else:
            bound_args=list(args);bound_kwargs=dict(kwargs)
            if kind=="call_function"and stable.startswith("torch.")and not bound_args:
                require("input"in bound_kwargs,"public_expression_input")
                bound_args=[bound_kwargs.pop("input")]
            require(bound_args and isinstance(bound_args[0],Proxy)and
                    all(p.node is bound_args[0].node for p in references),"public_expression_dynamic_metadata")
            parent=self.values[bound_args[0].node]
            logical,physical,proof=plan(parent.logical,kind,stable,self.static(tuple(bound_args)),
                                        self.static(bound_kwargs),resource_profile=self.profile,policy=self.policy)
        require(self.derived_count<self.limits.constants and
                self.derived_elements+physical.size<=self.limits.total_constant_elements,
                "public_expression_derived_capacity")
        if self.policy in ARITHMETIC_POLICIES:
            require(len(self.original)+self.derived_count<self.limits.constants and
                sum(v.size for v in self.original.values())+self.derived_elements+physical.size<=self.limits.total_constant_elements,
                "public_expression_total_state_capacity")
        return Selection(logical,physical,proof)
    def commit(self,node,selected):
        if selected is None:return
        self.values[node]=selected
        if selected.proof is not None:
            self.derived_count+=1;self.derived_elements+=selected.physical.size
    def divisor(self,proxy):
        require(proxy.node in self.values,"public_expression_divisor_not_closed")
        value=self.values[proxy.node].physical
        require(value.dtype.str=="<f8","public_expression_divisor_dtype")
        return value
    def unchanged(self):
        require(state_identity(_arrays(self.module),resource_profile=self.profile)==self.identity,
                "public_expression_tracing_changed_original_state")
