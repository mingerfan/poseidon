"""Known local FX fixtures, separate from production intake and Benchmark models."""
import torch
import torch.nn.functional as F
from fx_alias_source_plan import from_fx
from alias_source_plan import verify

APPROX=dict(relu=dict(method="polynomial",basis="power",domain=[-1.,1.],coefficients=[0.,.5,.25]))
CASES=(
 *[dict(id="relu_"+style+"_"+pattern,kind="relu",style=style,pattern=pattern) for style in ("method","functional","module") for pattern in ("base","shared","view")],
 *[dict(id="arithmetic_"+op+"_"+pattern,kind="arithmetic",operation=op,pattern=pattern) for op in ("add","sub","mul") for pattern in ("base","partial","disjoint")],
 dict(id="clone_dense_transpose",kind="clone"),
 dict(id="reshape_noncontiguous_copy",kind="reshape_copy"),
 dict(id="reshape_contiguous_view",kind="reshape_view"),
 dict(id="contiguous_alias_and_copy",kind="contiguous"),
 dict(id="repeated_mutation_and_prior_value",kind="repeated"),
 dict(id="split_view_writeback",kind="split"),
)
class Fixture(torch.nn.Module):
    def __init__(self,case):
        super().__init__();self.case=dict(case);self.activation=torch.nn.ReLU(inplace=True)
    def activation_call(self,x):
        style=self.case["style"]
        if style=="method":return x.relu_()
        if style=="functional":return F.relu(x,inplace=True)
        return self.activation(x)
    def forward(self,x,other=None):
        case=self.case;kind=case["kind"]
        if kind=="relu":
            pattern=case["pattern"]
            if pattern=="base":
                y=self.activation_call(x);return x,y
            if pattern=="shared":
                previous=x*1.5;alias=x;y=self.activation_call(alias);return previous+alias,x,y
            view=x.transpose(0,1);previous=x+.25;y=self.activation_call(view)
            return previous,x,view,y
        if kind=="arithmetic":
            method=case["operation"]+"_";pattern=case["pattern"]
            if pattern=="base":
                y=getattr(x,method)(.125);return x,y
            if pattern=="partial":
                previous=x.square();view=x[:,1:3]
                kwargs={"alpha":.5} if case["operation"] in ("add","sub") else {}
                y=getattr(view,method)(other,**kwargs);return previous,x,view,y
            previous=x.square();view=x[:2];tail=x[2:]
            y=getattr(view,method)(tail);return previous,x,y,tail
        if kind=="clone":
            t=x.transpose(0,1);c=t.clone();v=c.transpose(0,1);v.add_(.125);return x,t,c,v
        if kind=="reshape_copy":
            t=x.transpose(0,1);c=t.reshape(6);c.add_(.125);return x,t,c
        if kind=="reshape_view":
            v=x.reshape(6);v.mul_(.5);return x,v
        if kind=="contiguous":
            a=x.contiguous();b=x.transpose(0,1).contiguous();a.sub_(.125);b.add_(.25);return x,a,b
        if kind=="repeated":
            old=x.square();v=x[:,1:];x.add_(.125);v.mul_(.5);x.relu_();return old,x,v
        if kind=="split":
            a,b=x.split([1,2],dim=1);b.add_(.25);return x,a,b
        raise ValueError("Known fixture only")
def signature(case):
    shape=[4] if case.get("pattern")=="disjoint" else [2,3]
    items=[dict(name="x",shape=shape)]
    if case["kind"]=="arithmetic" and case["pattern"]=="partial":items.append(dict(name="other",shape=[2,2]))
    return items
def capture(case):
    fixture=Fixture(case).double().eval()
    traced=torch.fx.symbolic_trace(fixture)
    value,metadata=from_fx(traced,signature(case),APPROX)
    verify(value,expected_source_request_sha256=value["source_request_sha256"])
    return fixture,traced,value,metadata
