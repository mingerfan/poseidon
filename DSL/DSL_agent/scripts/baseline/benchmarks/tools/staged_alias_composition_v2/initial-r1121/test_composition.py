import copy,hashlib,json,unittest
from pathlib import Path
import numpy as np
from pytorch_alias_composition import plan,verify,SOURCE_FORMAT,AliasError,VALUE_OPERATORS
from benchmark_graph import digest
from benchmark_math import evaluate as lowered_reference
from operator_reference import evaluate as high_reference
from reference import evaluate as ordered_reference

PROFILE="general-tensors-v1"
RATIO=dict(method="newton",domain=[.1,2.],iterations=1)
MEASUREMENTS=[]
def node(op,inputs,output,attrs=None):
    return dict(id="n_"+output,op=op,inputs=inputs,attrs={} if attrs is None else attrs,output=output)
def call(op,inputs,output,attrs=None):
    return node("registered",inputs,output,dict(operator=op,parameters={} if attrs is None else attrs))
def base(shape,constants=None):
    return dict(format=SOURCE_FORMAT,inputs=[dict(name="x",shape=shape)],constants={"one":.125,**(constants or {})},nodes=[],outputs=[])
def probe_inputs(source):
    for kind in range(16):
        out={}
        for j,item in enumerate(source["inputs"]):
            count=int(np.prod(item["shape"]))
            if kind==0:a=np.zeros(count)
            elif kind==1:a=np.resize([-.2,.1,-.1,.2],count)
            elif kind==2:a=np.linspace(-.24,.24,count)
            else:a=np.random.default_rng(93000+100*kind+j).uniform(-.24,.24,count)
            out[item["name"]]=a.reshape(item["shape"])
        yield out

def context(group,variant):
    if group=="linear":
        s=base([2,3],dict(w=[[.25,-.125,.1875],[-.125,.25,.0625]],b=[.03125,-.015625]))
        s["nodes"]=[call("linear",["x","w","b"],"y")]
    elif group=="matmul":
        s=base([2,2],dict(w=[[.25,-.125],[.1875,.0625]]))
        s["nodes"]=[call("matmul",["x","w"],"y")]
    elif group=="conv1d":
        s=base([1,1,4],dict(w=[[[.25,-.125]]],b=[.03125]))
        s["nodes"]=[call("conv1d",["x","w","b"],"y",dict(stride=[1],padding=[0],dilation=[1],groups=1))]
    elif group=="conv2d":
        s=base([1,1,3,3],dict(w=[[[[.25,-.125],[.1875,.0625]]]],b=[-.015625]))
        s["nodes"]=[call("conv2d",["x","w","b"],"y",dict(stride=[1,1],padding=[0,0],dilation=[1,1],groups=1))]
    elif group=="reduction":
        s=base([2,3])
        s["nodes"]=[call("mean",["x"],"m",dict(axes=[1],keepdims=True)),
                    node("add",["x","m"],"y",dict(alpha=1.,inplace=False))]
    elif group=="join":
        s=base([2,2]);s["nodes"]=[node("square",["x"],"sq"),
            call("concat" if variant==0 else "stack",["x","sq"],"y",dict(axis=0 if variant<2 else 1))]
    elif group=="layer_norm":
        s=base([2,2],dict(gain=[1.,1.25],bias=[.03125,-.0625]))
        s["nodes"]=[call("layer_norm",["x","gain","bias"],"y",dict(normalized_shape=[2],eps=.1,approximation=RATIO))]
    elif group=="group_norm":
        s=base([1,2,2],dict(gain=[1.,1.25],bias=[.03125,-.0625]))
        s["nodes"]=[call("group_norm",["x","gain","bias"],"y",dict(groups=2,eps=.1,approximation=RATIO))]
    else:raise AssertionError(group)
    s["nodes"]+=[node("square",["y"],"previous"),node("alias",["y"],"alias_y")]
    if variant==0:
        s["nodes"]+=[node("add",["alias_y","one"],"mutated",dict(alpha=1.,inplace=True))]
    elif variant==1:
        s["nodes"]+=[node("slice",["alias_y"],"view_y",dict(index=[dict(kind="ellipsis"),dict(kind="slice",start=0,stop=1,step=1)])),
                    node("multiply",["view_y","one"],"mutated",dict(inplace=True))]
    else:
        s["nodes"]+=[node("clone",["alias_y"],"copied"),node("add",["copied","one"],"mutated",dict(alpha=1.,inplace=True))]
    s["outputs"]=[dict(name="current",value="y"),dict(name="previous",value="previous"),dict(name="result",value="mutated")]
    return s

class Composition(unittest.TestCase):
    def test_eight_operator_groups_three_alias_contexts(self):
        for group in ("linear","matmul","conv1d","conv2d","reduction","join","layer_norm","group_norm"):
            for variant in range(3):
                with self.subTest(group=group,variant=variant):
                    s=context(group,variant);p=plan(s,{},resource_profile=PROFILE)
                    maximum=0.;values=0
                    for inputs in probe_inputs(s):
                        saved={k:v.copy() for k,v in inputs.items()}
                        expected=ordered_reference(s,inputs)
                        for got in (high_reference(p["mathematical_graph"],inputs),
                                    lowered_reference(p["lowered_graph"],inputs,resource_profile=PROFILE)):
                            self.assertEqual(got.keys(),expected.keys())
                            for name,want in expected.items():
                                self.assertEqual(got[name].shape,want.shape)
                                np.testing.assert_allclose(got[name],want,atol=1e-12,rtol=1e-12)
                                maximum=max(maximum,float(np.max(np.abs(got[name]-want))))
                        for name in inputs:np.testing.assert_array_equal(inputs[name],saved[name])
                        values+=sum(v.size for v in expected.values())
                    self.assertTrue(any(e["rule"].startswith("registered_value:") for e in p["events"]))
                    self.assertFalse(p["production_intake_integrated"])
                    self.assertEqual(p["encrypted_status"],"unverified")
                    MEASUREMENTS.append(dict(group=group,variant=variant,probes=16,values=values,max_error=maximum,
                        mathematical_sha256=digest(p["mathematical_graph"]),lowered_sha256=digest(p["lowered_graph"])))
    def test_cipher_cipher_matmul_reads_updated_storage(self):
        s=base([2,2]);s["inputs"].append(dict(name="b",shape=[2,2]))
        s["nodes"]=[call("matmul",["x","b"],"before"),node("alias",["b"],"b_alias"),
                    node("add",["b_alias","one"],"b_new",dict(alpha=1.,inplace=True)),
                    call("matmul",["x","b"],"after")]
        s["outputs"]=[dict(name="before",value="before"),dict(name="after",value="after")]
        p=plan(s,{},resource_profile=PROFILE)
        for inputs in probe_inputs(s):
            want=ordered_reference(s,inputs)
            got=lowered_reference(p["lowered_graph"],inputs,resource_profile=PROFILE)
            for name in want:np.testing.assert_allclose(got[name],want[name],atol=1e-12,rtol=1e-12)
    def test_fresh_gather_not_view(self):
        s=base([2,2]);s["nodes"]=[call("public_gather",["x"],"selected",dict(axis=1,indices=[0])),
            node("add",["selected","one"],"mutated",dict(alpha=1.,inplace=True))]
        s["outputs"]=[dict(name="original",value="x"),dict(name="result",value="mutated")]
        p=plan(s,{},resource_profile=PROFILE)
        for inputs in probe_inputs(s):
            got=lowered_reference(p["lowered_graph"],inputs,resource_profile=PROFILE)
            np.testing.assert_array_equal(got["original"],inputs["x"])
            np.testing.assert_allclose(got["result"],inputs["x"][:,[0]]+.125,atol=1e-12,rtol=1e-12)
    def rejection(self,s,code="alias_call_rejected"):
        with self.assertRaises(AliasError) as e:plan(s,{},resource_profile=PROFILE)
        self.assertEqual(e.exception.code,code)
        return e.exception
    def test_alias_producing_calls_require_alias_path(self):
        for op in ("reshape","transpose","permute","slice","split","broadcast","dropout"):
            with self.subTest(op=op):
                self.assertNotIn(op,VALUE_OPERATORS)
                s=base([2,2]);s["nodes"]=[call(op,["x"],"y")];s["outputs"]=[dict(name="out",value="y")]
                self.rejection(s)
    def test_invalid_linear_shape(self):
        s=context("linear",0);s["constants"]["w"]=[[1.,2.]]
        self.rejection(s)
    def test_cipher_weights_rejected(self):
        s=base([2,2]);s["inputs"].append(dict(name="w",shape=[2,2]));s["nodes"]=[call("linear",["x","w"],"y")]
        s["outputs"]=[dict(name="out",value="y")]
        self.rejection(s,code="alias_graph_rejected")
    def test_no_callbacks_or_dynamic_registration(self):
        s=context("linear",0);s["nodes"][0]["attrs"]["parameters"]={"callback":"eval"}
        self.rejection(s)
        s=context("linear",0);s["nodes"][0]["attrs"]["operator"]="user_forward"
        self.rejection(s)
    def test_public_first_operand_rejected(self):
        s=base([2,2],dict(w=[[1.,0.],[0.,1.]]))
        s["nodes"]=[call("matmul",["w","x"],"y")];s["outputs"]=[dict(name="out",value="y")]
        self.rejection(s)
    def test_parameters_must_be_exact_registered_schema(self):
        s=context("reduction",0);s["nodes"][0]["attrs"]["parameters"]["keepdims"]="yes"
        self.rejection(s)
    def test_missing_approximation(self):
        s=context("layer_norm",0);del s["nodes"][0]["attrs"]["parameters"]["approximation"]
        self.rejection(s)
    def test_duplicate_source_ids_rejected(self):
        s=context("linear",0);s["nodes"][1]["id"]=s["nodes"][0]["id"]
        self.rejection(s,code="alias_source_rejected")
    def test_source_forward_reference_rejected(self):
        s=context("linear",0);s["nodes"][0]["inputs"][0]="previous"
        self.rejection(s,code="alias_source_rejected")
    def test_overlapping_write_still_rejected(self):
        s=base([2,2]);s["nodes"]=[node("slice",["x"],"a",dict(index=[dict(kind="slice",start=0,stop=1,step=1)])),
            node("slice",["x"],"b",dict(index=[dict(kind="slice",start=0,stop=1,step=1),dict(kind="slice",start=0,stop=1,step=1)])),
            node("add",["a","b"],"y",dict(alpha=1.,inplace=True))]
        s["outputs"]=[dict(name="out",value="y")]
        self.rejection(s,code="alias_overlap_rejected")
    def test_public_mutation_still_rejected(self):
        s=base([2,2],dict(w=[[1.,0.],[0.,1.]]))
        s["nodes"]=[node("add",["w","one"],"y",dict(alpha=1.,inplace=True))]
        s["outputs"]=[dict(name="out",value="x")]
        self.rejection(s,code="alias_public_mutation_rejected")
    def test_no_object_hooks(self):
        called=[]
        class Object:
            def __float__(self):called.append(True);return 1.
        s=context("linear",0);s["nodes"][0]["attrs"]["parameters"]=Object()
        self.rejection(s,code="alias_source_rejected");self.assertEqual(called,[])
    def test_binding_detects_weights_and_plan_edits(self):
        s=context("linear",0);p=plan(s,{},resource_profile=PROFILE)
        verify(p,expected_source_request_sha256=p["source_request_sha256"])
        changed=copy.deepcopy(p);changed["source_request"]["source"]["constants"]["w"][0][0]+=.125
        changed["binding"]=digest({k:v for k,v in changed.items() if k!="binding"})
        with self.assertRaises(AliasError):verify(changed,expected_source_request_sha256=p["source_request_sha256"])
        changed=copy.deepcopy(p);changed["events"][0]["shape"]=[999]
        changed["binding"]=digest({k:v for k,v in changed.items() if k!="binding"})
        with self.assertRaises(AliasError):verify(changed,expected_source_request_sha256=p["source_request_sha256"])
    def test_legacy_draft_kept(self):
        import alias_source_plan as old
        s=base([2,2]);s["format"]=old.SOURCE_FORMAT
        s["nodes"]=[node("square",["x"],"y"),node("add",["x","one"],"z",dict(alpha=1.,inplace=True))]
        s["outputs"]=[dict(name="out",value="y"),dict(name="mutated",value="z")]
        earlier=old.plan(s,{},resource_profile=PROFILE)
        current=copy.deepcopy(s);current["format"]=SOURCE_FORMAT
        later=plan(current,{},resource_profile=PROFILE)
        self.assertEqual(earlier["mathematical_graph"],later["mathematical_graph"])
        self.assertEqual(earlier["lowered_graph"],later["lowered_graph"])

if __name__=="__main__":unittest.main()
