"""Forward ABI bindings; real object identity and closed tree rejection controls."""
import copy, unittest
from source_forward_call import bind_call

def tensor(name):return dict(kind="tensor",name=name)
def literal(value):return dict(kind="literal",value=value)
def container(kind,items):return dict(kind=kind,items=items)
def spec(args,kwargs=None,names=("right","left")):
    return dict(inputs=[dict(name=n,shape=[2])for n in names],
                call=dict(format="poseidon-pytorch-call-v1",args=args,kwargs=kwargs or {}))

class ForwardCallTests(unittest.TestCase):
    def setUp(self):self.right=object();self.left=object();self.values=[self.right,self.left]
    def test_legacy_flat_order_is_preserved(self):
        m=dict(inputs=[dict(name="right"),dict(name="left")])
        args,kwargs=bind_call(m,self.values)
        self.assertIs(args[0],self.right);self.assertIs(args[1],self.left);self.assertEqual(kwargs,{})
    def test_declared_named_order_overrides_inventory_order(self):
        m=spec([tensor("left"),tensor("right")]);args,kwargs=bind_call(m,self.values)
        self.assertIs(args[0],self.left);self.assertIs(args[1],self.right)
    def test_dictionary_binding_preserves_tensor_identity(self):
        m=spec([container("dict",[["x",tensor("left")],["y",tensor("right")]])])
        args,_=bind_call(m,self.values);self.assertIs(args[0]["x"],self.left);self.assertIs(args[0]["y"],self.right)
    def test_nested_tuple_and_list_are_distinct(self):
        m=spec([container("tuple",[tensor("left"),container("list",[tensor("right")])])])
        args,_=bind_call(m,self.values)
        self.assertIs(type(args[0]),tuple);self.assertIs(type(args[0][1]),list)
        self.assertIs(args[0][0],self.left);self.assertIs(args[0][1][0],self.right)
    def test_keyword_and_positional_tensor_routes(self):
        m=spec([tensor("left")],{"other":tensor("right"),"gain":literal(.5)})
        args,kwargs=bind_call(m,self.values)
        self.assertIs(args[0],self.left);self.assertIs(kwargs["other"],self.right)
        self.assertEqual(kwargs["gain"],.5)
    def test_public_boolean_none_string_and_container_values_keep_types(self):
        m=spec([tensor("left"),tensor("right")],{
            "flag":literal(True),"optional":literal(None),"mode":literal("add"),
            "settings":container("list",[literal(.5),literal(-.25)])})
        args,kwargs=bind_call(m,self.values)
        self.assertIs(kwargs["flag"],True);self.assertIsNone(kwargs["optional"])
        self.assertEqual(kwargs["mode"],"add");self.assertIs(type(kwargs["settings"]),list)
    def test_unspecified_public_defaults_are_left_to_original_forward(self):
        args,kwargs=bind_call(spec([tensor("left"),tensor("right")]),self.values)
        self.assertEqual(kwargs,{})
    def test_public_containers_are_fresh_on_each_call(self):
        m=spec([tensor("left"),tensor("right")],{"settings":container("dict",[["gain",literal(.5)]])})
        before=copy.deepcopy(m);_,one=bind_call(m,self.values);one["settings"]["gain"]=3
        _,two=bind_call(m,self.values)
        self.assertEqual(two["settings"]["gain"],.5);self.assertEqual(m,before)
    def test_input_counts_and_duplicate_names_are_rejected(self):
        for m,values in ((spec([tensor("left"),tensor("right")]),[self.left]),
                         (dict(inputs=[dict(name="left"),dict(name="left")]),self.values),
                         (dict(inputs=[]),[])):
            with self.subTest(m=m),self.assertRaises(ValueError):bind_call(m,values)
    def test_missing_duplicate_or_unknown_tensor_binding_rejected(self):
        for args in ([tensor("left")],[tensor("left"),tensor("left")],[tensor("left"),tensor("unknown")]):
            with self.subTest(args=args),self.assertRaises(ValueError):bind_call(spec(args),self.values)
    def test_extra_call_fields_and_callbacks_are_rejected(self):
        m=spec([tensor("left"),tensor("right")]);m["call"]["callback"]="execute"
        with self.assertRaises(ValueError):bind_call(m,self.values)
        m=spec([tensor("left"),tensor("right")],{"callback":literal(lambda:1)})
        with self.assertRaises(ValueError):bind_call(m,self.values)
    def test_outside_literal_budget_is_rejected(self):
        for value in (float("nan"),float("inf"),1025.,2**53+1):
            m=spec([tensor("left"),tensor("right")],{"gain":literal(value)})
            with self.subTest(value=value),self.assertRaises(ValueError):bind_call(m,self.values)
    def test_source_objects_are_not_coerced(self):
        class NeverArray:
            def __array__(self):raise AssertionError("Unexpected coercion")
        left=NeverArray();right=NeverArray()
        args,_=bind_call(spec([tensor("left"),tensor("right")]),[right,left])
        self.assertIs(args[0],left);self.assertIs(args[1],right)
    def test_mutation_tracking_can_still_use_named_tensor_order(self):
        named=[["right"],["left"]]
        args,_=bind_call(spec([container("dict",[["x",tensor("left")],["y",tensor("right")]])]),named)
        args[0]["x"].append("mutated")
        self.assertEqual(named,[["right"],["left","mutated"]])

if __name__=="__main__":unittest.main()
