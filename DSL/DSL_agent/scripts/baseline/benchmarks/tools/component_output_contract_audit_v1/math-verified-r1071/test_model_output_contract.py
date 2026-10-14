import copy,math,unittest
import numpy as np
import torch
from benchmark_graph import FORMAT,CAPACITY_FORMAT,digest
from model_resources import identity
from model_output_contract import (build,verify,capture_tree,restore,bind_delivery,
 verify_delivery,structure_execution,OutputError,tensor_leaves,snapshot,implementation,TREE_DEPTH,SNAPSHOT_DEPTH)
from pytorch_source_contract import output_tree as legacy_capture,restore_outputs as legacy_restore

SOURCE="a"*64
REQUEST="b"*64
BUNDLE="c"*64
VALIDATION=dict(level="numerical",encrypted_execution=True,numerically_validated=True)
def model(shape=(2,),names=("out",),profile=None):
    value=dict(format=CAPACITY_FORMAT if profile else FORMAT,id="output_example",
        inputs=[dict(name="x",shape=list(shape))],constants={},
        nodes=[dict(id="square",op="square",inputs=["x"],attrs={},outputs=["y"])],
        outputs=[dict(name=n,value="y") for n in names])
    if profile:value["resource_profile"]=profile
    return value
def tensor(name="out",scalar=False):
    return dict(kind="tensor",name=name,**({"shape":[]} if scalar else {}))
def contract(tree=None,graph=None):
    return build(graph or model(),tree or tensor(),source_binding=SOURCE)
def delivery(contract_value=None,validation=None):
    c=contract_value or contract()
    return bind_delivery(c,request_id=REQUEST,bundle_binding=BUNDLE,package_validation=validation or VALIDATION)
def report(outputs=None,validation=None):
    return dict(format="poseidon-local-execution-v1",status="executed",backend="poseidon_cpu",
        request_id=REQUEST,bundle_binding=BUNDLE,package_validation=validation or copy.deepcopy(VALIDATION),
        encrypted_execution=True,executor_secret_key=False,numerically_validated_on_supplied_inputs=False,
        application_accuracy_certified=False,samples=1,outputs=outputs or [dict(out=[.125,-.25])],
        agent_calls=0,compiler_calls=0,validation_scope="Supplied inputs executed; no reference comparison requested")

class OutputsTests(unittest.TestCase):
    def reject(self,code,fn,*args,**kwargs):
        with self.assertRaises(OutputError) as cm:fn(*args,**kwargs)
        self.assertEqual(cm.exception.code,code)
        p=cm.exception.feedback()
        self.assertEqual(p["retry_policy"],"never_automatic")
        self.assertEqual(p["layer"],"delivery")
        self.assertNotIn("private_source",str(p))
    def test_tensor_contract_uses_graph_shape(self):
        c=contract();self.assertEqual(c["signature"],[dict(name="out",abi_shape=[2],logical_shape=[2],dtype="float64")])
        verify(c,expected_binding=c["binding"],model=model(),source_binding=SOURCE)
    def test_static_none_has_no_ciphertext_output(self):
        tree=dict(kind="tuple",items=[tensor(),dict(kind="none")]);c=contract(tree)
        self.assertEqual(len(c["signature"]),1)
        self.assertEqual(restore(c,dict(out=[.1,.2]),expected_binding=c["binding"]),([.1,.2],None))
    def test_nested_dictionary_list_and_tuple(self):
        tree=dict(kind="dict",items=[dict(key="logits",value=tensor()),
            dict(key="cache",value=dict(kind="tuple",items=[dict(kind="list",items=[tensor("state"),dict(kind="none")])]))])
        c=contract(tree,model(names=("out","state")))
        actual=restore(c,dict(out=[.1,.2],state=[.3,.4]),expected_binding=c["binding"])
        self.assertEqual(actual,{"logits":[.1,.2],"cache":([[.3,.4],None],)})
    def test_scalar_tensor_abi_marker(self):
        c=contract(tensor(scalar=True),model(shape=(1,)))
        self.assertEqual(restore(c,dict(out=[.125]),expected_binding=c["binding"]),.125)
        self.assertEqual(c["signature"][0]["logical_shape"],[])
    def test_scalar_marker_requires_one_element(self):
        self.reject("output_signature_rejected",build,model(),tensor(scalar=True),source_binding=SOURCE)
    def test_none_only_does_not_become_numerical_success(self):
        self.reject("output_signature_rejected",build,model(),dict(kind="none"),source_binding=SOURCE)
    def test_named_output_order_is_bound(self):
        graph=model(names=("left","right"))
        self.reject("output_signature_rejected",build,graph,
                    dict(kind="tuple",items=[tensor("right"),tensor("left")]),source_binding=SOURCE)
    def test_duplicate_tensor_leaf_is_rejected(self):
        self.reject("output_signature_rejected",build,model(names=("left","right")),
                    dict(kind="tuple",items=[tensor("left"),tensor("left")]),source_binding=SOURCE)
    def test_unknown_tensor_name_is_rejected(self):
        self.reject("output_signature_rejected",build,model(),tensor("unknown"),source_binding=SOURCE)
    def test_duplicate_dictionary_keys_are_rejected(self):
        tree=dict(kind="dict",items=[dict(key="same",value=tensor()),dict(key="same",value=dict(kind="none"))])
        self.reject("output_structure_rejected",build,model(),tree,source_binding=SOURCE)
    def test_unknown_node_fields_are_rejected(self):
        tree=dict(tensor(),private_source="do not expose this")
        self.reject("output_structure_rejected",build,model(),tree,source_binding=SOURCE)
    def test_unknown_static_scalar_leaf_is_rejected(self):
        self.reject("output_structure_rejected",build,model(),dict(kind="constant",value=0),source_binding=SOURCE)
    def test_empty_containers_are_rejected(self):
        for kind in ("list","tuple","dict"):
            with self.subTest(kind=kind):
                self.reject("output_structure_rejected",build,model(),dict(kind=kind,items=[]),source_binding=SOURCE)
    def test_structure_depth_boundary(self):
        tree=tensor()
        for _ in range(32):tree=dict(kind="tuple",items=[tree])
        c=contract(tree);verify(c,expected_binding=c["binding"])
        tree=dict(kind="tuple",items=[tree])
        self.reject("output_structure_rejected",build,model(),tree,source_binding=SOURCE)
    def test_dictionary_depth_works_through_delivery_envelopes(self):
        tree=tensor();expected=[.125,-.25]
        for _ in range(TREE_DEPTH):
            tree=dict(kind="dict",items=[dict(key="value",value=tree)])
            expected={"value":expected}
        c=contract(tree);d=delivery(c)
        verify_delivery(d,expected_binding=d["binding"],model=model(),source_binding=SOURCE)
        actual=structure_execution(d,report(),expected_binding=d["binding"],model=model(),source_binding=SOURCE)
        self.assertEqual(actual["structured_outputs"],[expected])
        tree=dict(kind="dict",items=[dict(key="value",value=tree)])
        self.reject("output_structure_rejected",build,model(),tree,source_binding=SOURCE)
    def test_generic_snapshot_depth_remains_bounded(self):
        value=0
        for _ in range(SNAPSHOT_DEPTH):value=[value]
        self.assertEqual(snapshot(value),value)
        self.reject("output_source_rejected",snapshot,[value])
    def test_logical_dimensions_require_integer_types(self):
        for shape in ([True],[1.0],["1"]):
            c=contract(graph=model(shape=(1,)));c["signature"][0]["logical_shape"]=shape
            c["binding"]=digest({k:v for k,v in c.items() if k!="binding"})
            self.reject("output_signature_rejected",verify,c,expected_binding=c["binding"])
    def test_validation_fields_require_actual_booleans(self):
        for level,flag in (("compiled",False),("numerical",True)):
            for key in ("encrypted_execution","numerically_validated"):
                for bad in (int(flag),float(flag),str(flag)):
                    v=dict(level=level,encrypted_execution=flag,numerically_validated=flag)
                    v[key]=bad
                    self.reject("output_execution_identity_rejected",bind_delivery,contract(),
                                request_id=REQUEST,bundle_binding=BUNDLE,package_validation=v)
    def test_resigned_validation_does_not_allow_integer_booleans(self):
        d=delivery()
        d["package_validation"]["numerically_validated"]=1
        d["binding"]=digest({k:v for k,v in d.items() if k!="binding"})
        self.reject("output_execution_identity_rejected",verify_delivery,d,expected_binding=d["binding"])
    def test_execution_validation_does_not_allow_integer_booleans(self):
        d=delivery()
        for key in ("encrypted_execution","numerically_validated"):
            p=report();p["package_validation"]=copy.deepcopy(VALIDATION);p["package_validation"][key]=1
            self.reject("output_execution_identity_rejected",structure_execution,d,p,expected_binding=d["binding"])
    def test_malformed_delivery_contract_has_structured_failure(self):
        for value in (None,{},[]):
            self.reject("output_binding_rejected",bind_delivery,value,request_id=REQUEST,bundle_binding=BUNDLE,
                        package_validation=VALIDATION)
    def test_source_binding_is_required(self):
        self.reject("output_binding_rejected",build,model(),tensor(),source_binding="private_source")
    def test_model_content_changes_binding(self):
        a=contract();g=model();g["nodes"][0]["op"]="negate"
        b=contract(graph=g);self.assertNotEqual(a["binding"],b["binding"])
        self.reject("output_binding_rejected",verify,a,expected_binding=a["binding"],model=g)
    def test_source_content_changes_binding(self):
        a=contract();b=build(model(),tensor(),source_binding="d"*64)
        self.assertNotEqual(a["binding"],b["binding"])
        self.reject("output_binding_rejected",verify,a,expected_binding=a["binding"],source_binding="d"*64)
    def test_resigned_contract_is_rejected_by_trusted_binding(self):
        c=contract();trusted=c["binding"];c["source_binding"]="d"*64
        c["binding"]=digest({k:v for k,v in c.items() if k!="binding"})
        self.reject("output_binding_rejected",verify,c,expected_binding=trusted)
    def test_resigned_bad_signature_is_rejected(self):
        c=contract();c["signature"][0]["abi_shape"]=[1]
        c["binding"]=digest({k:v for k,v in c.items() if k!="binding"})
        self.reject("output_signature_rejected",verify,c,expected_binding=c["binding"])
    def test_wrong_profile_is_rejected(self):
        c=contract();c["resource_profile"]=identity("general-tensors-v1")
        c["binding"]=digest({k:v for k,v in c.items() if k!="binding"})
        self.reject("output_binding_rejected",verify,c,expected_binding=c["binding"],model=model())
    def test_rank32_shape_is_preserved(self):
        shape=(1,)*31+(2,);graph=model(shape=shape,profile="general-tensors-v1");c=contract(graph=graph)
        value=np.asarray([.1,.2],dtype=np.float64).reshape(shape)
        self.assertEqual(restore(c,dict(out=value),expected_binding=c["binding"]),value.tolist())
    def test_wrong_output_names_are_rejected(self):
        c=contract();self.reject("output_values_rejected",restore,c,dict(private_source=[1,2]),expected_binding=c["binding"])
    def test_wrong_shape_is_rejected(self):
        c=contract()
        for values in ([.1],[[.1,.2]],[.1,.2,.3]):
            with self.subTest(shape=str(np.shape(values))):
                self.reject("output_values_rejected",restore,c,dict(out=values),expected_binding=c["binding"])
    def test_nonfinite_and_boolean_values_are_rejected(self):
        c=contract()
        for x in (float("nan"),float("inf"),-float("inf"),True):
            self.reject("output_values_rejected",restore,c,dict(out=[x,.125]),expected_binding=c["binding"])
    def test_large_integer_precision_is_rejected(self):
        c=contract();self.reject("output_values_rejected",restore,c,dict(out=[2**53+1,0]),expected_binding=c["binding"])
    def test_ndarray_precision_is_explicit(self):
        c=contract()
        for dtype in (np.float32,np.int64,np.bool_):
            self.reject("output_values_rejected",restore,c,dict(out=np.asarray([0,1],dtype=dtype)),expected_binding=c["binding"])
        self.assertEqual(restore(c,dict(out=np.asarray([.1,.2],dtype=np.float64)),expected_binding=c["binding"]),[.1,.2])
    def test_conversion_hooks_are_never_called(self):
        class Unsafe:
            def __array__(self,*a,**k):raise AssertionError("Array hook")
            def __deepcopy__(self,memo):raise AssertionError("Copy hook")
        c=contract()
        self.reject("output_values_rejected",restore,c,dict(out=Unsafe()),expected_binding=c["binding"])
        self.reject("output_source_rejected",snapshot,dict(private_source=Unsafe()))
    def test_container_subclass_hooks_are_never_called(self):
        class Unsafe(list):
            def __iter__(self):raise AssertionError("Iteration hook")
        self.reject("output_structure_rejected",capture_tree,Unsafe(),lambda v:False,["out"])
    def test_cycles_are_bounded(self):
        value={};value["private_source"]=value
        self.reject("output_source_rejected",snapshot,value)
    def test_restored_values_are_independent_snapshots(self):
        c=contract();values=dict(out=[.1,.2]);restored=restore(c,values,expected_binding=c["binding"])
        restored[0]=1.;self.assertEqual(values,dict(out=[.1,.2]))
    def test_capture_and_legacy_contract_agree_on_registered_old_trees(self):
        leaf=torch.tensor([.1,.2],dtype=torch.float64)
        for value in (leaf,(leaf,),[leaf],{"value":leaf},{"nested":([leaf],)}):
            a,ta=legacy_capture(value,lambda v:type(v) is torch.Tensor,["out"])
            b,tb=capture_tree(value,lambda v:type(v) is torch.Tensor,["out"])
            self.assertEqual(ta,tb);self.assertIs(a[0],b[0])
            c=contract(tb);self.assertEqual(restore(c,dict(out=[.1,.2]),expected_binding=c["binding"]),legacy_restore(dict(out=[.1,.2]),ta))
    def test_capture_supports_real_tensor_with_static_none(self):
        x=torch.tensor([.1,.2],dtype=torch.float64);value=(x,None)
        with self.assertRaises(ValueError):legacy_capture(value,lambda v:type(v) is torch.Tensor,["out"])
        leaves,tree=capture_tree(value,lambda v:type(v) is torch.Tensor,["out"])
        self.assertEqual(tree,dict(kind="tuple",items=[tensor(),dict(kind="none")]))
        self.assertIs(leaves[0],x)
    def test_capture_scalar_tensor_matches_old_schema(self):
        value=torch.tensor(.125,dtype=torch.float64)
        a,old=legacy_capture(value,lambda v:type(v) is torch.Tensor,["out"],scalar_tensor=lambda v:v.ndim==0)
        b,new=capture_tree(value,lambda v:type(v) is torch.Tensor,["out"],scalar_tensor=lambda v:v.ndim==0)
        self.assertEqual(old,new);self.assertIs(a[0],b[0])
    def test_delivery_binding_and_unchanged_validation(self):
        d=delivery();self.assertEqual(d["package_validation"],VALIDATION)
        verify_delivery(d,expected_binding=d["binding"],model=model(),source_binding=SOURCE)
    def test_delivery_validation_level_is_explicit(self):
        v=dict(level="compiled",encrypted_execution=False,numerically_validated=False);d=delivery(validation=v)
        packet=report(validation=v);wrapped=structure_execution(d,packet,expected_binding=d["binding"])
        self.assertEqual(wrapped["execution"]["package_validation"],v)
        self.assertFalse(wrapped["execution"]["numerically_validated_on_supplied_inputs"])
        self.assertNotIn("numerically_validated",wrapped)
    def test_unknown_grade_is_rejected(self):
        self.reject("output_execution_identity_rejected",delivery,validation=dict(level="compiled",encrypted_execution=True,numerically_validated=True))
    def test_resigned_delivery_is_rejected(self):
        d=delivery();trusted=d["binding"];d["bundle_binding"]="d"*64
        d["binding"]=digest({k:v for k,v in d.items() if k!="binding"})
        self.reject("output_binding_rejected",verify_delivery,d,expected_binding=trusted)
    def test_existing_execution_record_is_preserved(self):
        d=delivery();packet=report();before=copy.deepcopy(packet)
        result=structure_execution(d,packet,expected_binding=d["binding"])
        self.assertEqual(result["execution"],before);self.assertEqual(packet,before)
        self.assertEqual(result["structured_outputs"],[[.125,-.25]])
    def test_none_restored_after_existing_named_output_execution(self):
        c=contract(dict(kind="tuple",items=[tensor(),dict(kind="none")]));d=delivery(c)
        result=structure_execution(d,report(),expected_binding=d["binding"])
        self.assertEqual(result["structured_outputs"],[([.125,-.25],None)])
    def test_package_request_and_validation_drift_is_rejected(self):
        d=delivery()
        for key,value in (("request_id","d"*64),("bundle_binding","d"*64),("backend","seal_cpu"),
                          ("package_validation",dict(level="compiled",encrypted_execution=False,numerically_validated=False))):
            packet=report();packet[key]=value
            self.reject("output_execution_identity_rejected",structure_execution,d,packet,expected_binding=d["binding"])
    def test_plaintext_result_cannot_be_delivered_as_encrypted(self):
        d=delivery();packet=report();packet["encrypted_execution"]=False
        self.reject("output_execution_identity_rejected",structure_execution,d,packet,expected_binding=d["binding"])
    def test_failure_terminals_remain_failures(self):
        d=delivery()
        for status in ("validation_failed","service_error","cancelled","timed_out"):
            packet=dict(status=status,failure=dict(code="original_failure",layer="compiler"),encrypted_execution=False)
            result=structure_execution(d,packet,expected_binding=d["binding"])
            self.assertEqual(result["status"],status);self.assertEqual(result["execution"],packet)
            self.assertNotIn("structured_outputs",result)
    def test_sample_bounds_and_named_result_count(self):
        d=delivery()
        for n in (0,17):
            packet=report();packet["samples"]=n;packet["outputs"]=[dict(out=[.1,.2])]*max(1,n)
            self.reject("output_values_rejected",structure_execution,d,packet,expected_binding=d["binding"])
        packet=report();packet["samples"]=2
        self.reject("output_values_rejected",structure_execution,d,packet,expected_binding=d["binding"])
    def test_16_samples_can_be_delivered_without_new_validation_claims(self):
        d=delivery();packet=report();packet["samples"]=16
        packet["outputs"]=[dict(out=[.125,-.25]) for _ in range(16)]
        result=structure_execution(d,packet,expected_binding=d["binding"])
        self.assertEqual(len(result["structured_outputs"]),16)
        self.assertFalse(result["execution"]["numerically_validated_on_supplied_inputs"])
    def test_metadata_large_modulus_integers_are_not_float_cast(self):
        d=delivery();packet=report();packet["public_moduli"]=[2**60-93]
        result=structure_execution(d,packet,expected_binding=d["binding"])
        self.assertEqual(result["execution"]["public_moduli"],[2**60-93])
        self.assertIs(type(result["execution"]["public_moduli"][0]),int)

if __name__=="__main__":unittest.main()
