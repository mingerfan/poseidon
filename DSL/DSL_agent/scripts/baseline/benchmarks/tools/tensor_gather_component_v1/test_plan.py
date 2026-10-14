"""Pure mathematical checks; pinned Torch and encrypted acceptance are separate."""
import copy,itertools,math,unittest
from tensor_gather_plan import plan,check_plan,index_tensor

def build(shape, function, prefix=()):
    if not shape:
        return function(prefix)
    return [build(shape[1:],function,prefix+(i,)) for i in range(shape[0])]

def cell(tensor, coordinate):
    for i in coordinate:
        tensor=tensor[i]
    return tensor

def flatten(value):
    if type(value) is list:
        return [x for child in value for x in flatten(child)]
    return [value]

class TensorGatherTests(unittest.TestCase):
    def assert_context(self, shape, index, axis, resource="legacy-v1"):
        result=plan(shape,index,axis,resource_profile=resource)
        target=index_tensor(index)[0]
        # Oracle reads original tensor coordinates, not the flattened plan.
        source=build(shape,lambda q:sum((i+1)*100**j for j,i in enumerate(q))+.375)
        expected=[]
        for coordinate in itertools.product(*(range(v) for v in target)):
            origin=list(coordinate);origin[axis%len(shape)]=cell(index,coordinate)
            expected.append(cell(source,origin))
        values=flatten(source)
        actual=[values[i]for i in result["steps"][1]["attrs"]["indices"]]
        self.assertEqual(actual,expected)
        self.assertEqual(result["result_shape"],list(target))
        self.assertFalse(result["backend_or_layout_claimed"])
        self.assertEqual(check_plan(result,shape,index,axis,resource_profile=resource),result)

    def test_vector_reordered_and_repeated_indices(self):
        self.assert_context([5],[4,0,4,2],0)
    def test_each_row_has_different_indices(self):
        self.assert_context([2,4],[[3,0,3],[1,2,0]],1)
    def test_middle_axis_with_smaller_nonindexed_dimensions(self):
        self.assert_context([2,3,2],[[[2],[0]],[[1],[2]]],-2)
    def test_rank32_coordinates(self):
        shape=[1]*30+[2,3];index=[[2,0],[1,2]]
        for _ in range(30):index=[index]
        self.assert_context(shape,index,-1,"general-tensors-v1")
    def test_identity_mapping_is_retained(self):
        self.assert_context([2,3],[[0,1,2],[0,1,2]],1)
    def test_public_index_duplicate_contributions_preserved(self):
        result=plan([2,3],[[1,1],[2,2]],1)
        self.assertEqual(result["steps"][1]["attrs"]["indices"],[1,1,5,5])
    def test_old_one_dimensional_index_select_is_different(self):
        result=plan([2,4],[[3,0],[1,2]],1)
        self.assertEqual(result["steps"][1]["attrs"]["indices"],[3,0,5,6])
        self.assertNotEqual(result["steps"][1]["attrs"]["indices"],[3,0,7,4])
    def test_boolean_and_float_indices_rejected(self):
        for data in ([True,False],[0.0,1.0]):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):plan([2],data,0)
    def test_index_dtype_is_exact(self):
        for dtype in ("int32","bool","float64",None):
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError):plan([2],[0,1],0,index_dtype=dtype)
    def test_negative_or_out_of_range_indices_rejected(self):
        for data in ([-1],[2],[2**63]):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):plan([2],data,0)
    def test_empty_and_ragged_indices_rejected(self):
        for data in ([],[[],[]],[[0],[1,0]]):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):plan([2,2],data,1)
    def test_wrong_rank_rejected(self):
        with self.assertRaises(ValueError):plan([2,3],[0,1],1)
    def test_nonindexed_axis_cannot_expand(self):
        with self.assertRaises(ValueError):plan([2,3],[[0],[1],[2]],1)
    def test_invalid_or_dynamic_axis_rejected(self):
        for axis in (True,2,-3,"1",None):
            with self.subTest(axis=axis):
                with self.assertRaises(ValueError):plan([2,3],[[0],[1]],axis)
    def test_profile_rank_and_tensor_limits_remain(self):
        with self.assertRaises(ValueError):plan([1]*5,[[[[[0]]]]],0)
        with self.assertRaises(ValueError):plan([257],[0],0)
    def test_deep_or_cyclic_indices_fail_as_value_errors(self):
        deep=0
        for _ in range(200):deep=[deep]
        cyclic=[];cyclic.append(cyclic)
        for data in (deep,cyclic):
            with self.assertRaises(ValueError):plan([1],data,0)
    def test_expansion_minimum_keeps_existing_node_budget(self):
        with self.assertRaises(ValueError):plan([1],[0]*64,0)
    def test_rectangularity_cannot_mix_leaf_depths(self):
        with self.assertRaises(ValueError):index_tensor([0,[1]])
    def test_tampered_offsets_shape_and_binding_rejected(self):
        original=plan([2,3],[[0,1],[2,0]],1)
        for field in ("offset","shape","binding"):
            changed=copy.deepcopy(original)
            if field=="offset":changed["steps"][1]["attrs"]["indices"][0]=1
            if field=="shape":changed["result_shape"]=[4]
            if field=="binding":changed["binding"]="0"*64
            with self.assertRaises(ValueError):check_plan(changed,[2,3],[[0,1],[2,0]],1)
if __name__=="__main__":unittest.main(verbosity=2)
