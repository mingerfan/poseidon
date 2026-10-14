"""Tensor-axis guidance and application feedback; no model API or FHE calls."""
import copy,itertools,json,tempfile,unittest
from pathlib import Path
from benchmark_graph import CAPACITY_FORMAT,digest
from benchmark_suite import Builder
from component_contract import prepare_task,reconstruct_request,validate_program
from unified_graph_contract import validate_request,static_repair_hint
from unified_tensor_axis_guidance import specification,repair_hint,MAX_BYTES
from application_diagnostics import from_result,repeat_key,attach

def request(graph,version="explicit-v12"):
    g=copy.deepcopy(graph)
    g.update(format=CAPACITY_FORMAT,resource_profile="general-tensors-v1")
    return prepare_task(g,dict(generation_guidance=version,chunk_period=4))
def reduction(shape,axes,keepdims=False,operator="sum"):
    b=Builder([shape])
    return b.finish(b.node(operator,["input0"],axes=axes,keepdims=keepdims))
def annotations(r):
    return r["generation_guidance"]["poseidon_expression_rules"]["tensor_axis_annotations"]
def candidate(r,source=None):
    if source is None:
        names=r["generation_guidance"]["ordered_parameters"]
        source=r["generation_guidance"]["entry_header"]+"\n    return "+names[0]+"\n"
    return dict(schema=1,request_id=r["request_id"],hecate_source=source)

class TensorAxisTests(unittest.TestCase):
    def test_last_axis_keeps_head_and_row_groups(self):
        r=request(reduction((2,2,2),[-1]))
        row=annotations(r)["reductions"][0]
        self.assertEqual(row["reduced_axes"],[2])
        self.assertEqual(row["preserved_axes"],[0,1])
        fibers={x["output_index"]:x for x in row["logical_fibers"]}
        self.assertEqual(fibers[0]["input_indices"],[0,1])
        self.assertEqual(fibers[3]["input_indices"],[6,7])
        self.assertTrue(all(x["complete"] for x in fibers.values()))

    def test_non_contiguous_axis_preserves_distinct_columns(self):
        row=annotations(request(reduction((2,2,2,2),[2])))["reductions"][0]
        fibers={x["output_index"]:x["input_indices"] for x in row["logical_fibers"]}
        self.assertEqual(fibers[0],[0,2])
        self.assertEqual(fibers[7],[13,15])
        self.assertEqual(row["preserved_axes"],[0,1,3])

    def test_multi_axis_mean_with_and_without_keepdims(self):
        for keep in (False,True):
            row=annotations(request(reduction((2,3,2),[0,-1],keep,"mean")))["reductions"][0]
            fibers={x["output_index"]:x["input_indices"] for x in row["logical_fibers"]}
            self.assertEqual(fibers[0],[0,1,6,7])
            self.assertEqual(fibers[2],[4,5,10,11])
            self.assertEqual(row["member_count"],4)
            self.assertEqual(row["output_shape"],[1,3,1] if keep else [3])
            self.assertEqual(row["operator"],"mean")

    def test_full_reduction_singleton_output(self):
        row=annotations(request(reduction((2,2),[0,1])))["reductions"][0]
        self.assertEqual(row["preserved_axes"],[])
        self.assertEqual(row["output_shape"],[1])
        self.assertEqual(row["logical_fibers"][0]["input_indices"],[0,1,2,3])

    def test_large_fiber_marks_sample_as_incomplete(self):
        row=annotations(request(reduction((2,12),[1])))["reductions"][0]
        self.assertEqual(row["member_count"],12)
        for f in row["logical_fibers"]:
            self.assertFalse(f["complete"])
            self.assertEqual(f["listed_members"],8)
            self.assertEqual(f["total_members"],12)
            self.assertEqual(f["input_indices"][-1],f["output_index"]*12+11)

    def test_rank_32_and_tail_chunk_coordinates(self):
        shape=(1,)*30+(2,3)
        r=request(reduction(shape,[-1],True))
        a=annotations(r)
        self.assertEqual(len(a["reductions"][0]["input_shape"]),32)
        self.assertEqual([(x["logical_offset"],x["valid_elements"],x["padded_slots"])
                         for x in a["input_chunks"]],[(0,4,0),(4,2,2)])
        final=a["input_chunks"][-1]["sampled_input_coordinates"][-1]
        self.assertEqual(final["logical_index"],5)
        self.assertEqual(final["coordinate"],[0]*30+[1,2])

    def test_reshape_preserves_indices_and_permute_moves_coordinates(self):
        b=Builder([(2,3)])
        a=b.node("reshape",["input0"],shape=[3,2])
        y=b.node("permute",[a],dims=[1,0])
        rows=annotations(request(b.finish(y)))["transforms"]
        self.assertTrue(all(x["output_index"]==x["input_index"] for x in rows[0]["sampled_index_map"]))
        mapping={x["output_index"]:x["input_index"] for x in rows[1]["sampled_index_map"]}
        self.assertEqual(mapping[0],0)
        self.assertEqual(mapping[5],5)
        self.assertEqual(mapping[1],2)

    def test_transpose_negative_axes(self):
        b=Builder([(2,3)])
        y=b.node("transpose",["input0"],dim0=-2,dim1=-1)
        row=annotations(request(b.finish(y)))["transforms"][0]
        mapping={x["output_index"]:x["input_index"] for x in row["sampled_index_map"]}
        self.assertEqual(mapping[1],3)
        self.assertEqual(mapping[3],4)

    def test_batched_linear_and_broadcast_groups(self):
        b=Builder([(2,2)])
        h=b.node("linear",["input0",b.const([[.1,.2],[.3,.4]])])
        mean=b.node("mean",[h],axes=[-1],keepdims=True)
        y=b.node("multiply",[h,mean])
        r=request(b.finish(y));a=annotations(r)
        self.assertEqual(a["linears"][0]["preserved_input_axes"],[0])
        self.assertEqual(a["linears"][0]["weight_feature_axis"],1)
        self.assertEqual(a["broadcasts"][0]["operand_shapes"],[[2,2],[2,1]])
        self.assertIn("linear_contraction_groups",r["generation_guidance"]["poseidon_expression_rules"]["rules"])

    def test_examples_and_bytes_bounded_without_losing_partition_counts(self):
        b=Builder([(2,2)])
        outputs=[]
        for i in range(48):
            outputs.append(b.node("mean",["input0"],axes=[i%2],keepdims=True))
        r=request(b.finish(outputs[-1]))
        ctx=r["generation_guidance"]["poseidon_expression_rules"]
        self.assertEqual(ctx["tensor_axis_annotations"]["reduction_count"],48)
        self.assertLessEqual(len(ctx["tensor_axis_annotations"]["reductions"]),12)
        self.assertLessEqual(len(json.dumps(ctx,sort_keys=True).encode()),MAX_BYTES)
        self.assertIn("segmented_logical_reduction",ctx["rules"])

    def test_private_data_never_enters_public_context_or_feedback(self):
        r=request(reduction((2,2),[-1]))
        r["private_inputs"]="PRIVATE_ARRAY";r["reference"]="PRIVATE_REFERENCE"
        text=json.dumps(specification(r))
        text+=repair_hint(r,candidate(r)["hecate_source"],"PRIVATE_DIAG /home/secret","numerical_comparison")
        for s in ("PRIVATE_ARRAY","PRIVATE_REFERENCE","PRIVATE_DIAG","/home/secret"):
            self.assertNotIn(s,text)

    def test_graph_id_does_not_select_axis_rules(self):
        r=request(reduction((2,2),[0]))
        expected=specification(r)
        r["model"]["id"]="never_seen_family"
        self.assertEqual(specification(r),expected)

    def test_version_explicit_and_rehash_cannot_tamper_with_members(self):
        g=reduction((2,2),[0])
        old,new=request(g,"explicit-v11"),request(g)
        self.assertNotEqual(old["request_id"],new["request_id"])
        self.assertEqual(reconstruct_request(old),old)
        self.assertEqual(reconstruct_request(new),new)
        new["generation_guidance"]["poseidon_expression_rules"]["tensor_axis_annotations"]["reductions"][0]["logical_fibers"][0]["input_indices"]=[0,1]
        new["request_id"]=digest({k:v for k,v in new.items() if k!="request_id"})
        with self.assertRaises(ValueError):
            reconstruct_request(new)

    def test_invalid_abi_and_unbound_chunk_range_rejected(self):
        r=request(reduction((2,2),[0]))
        r["layout"]["execution_abi"]="unknown"
        with self.assertRaises(ValueError):specification(r)
        r=request(reduction((2,2),[0]));r["layout"]["inputs"][0]["offset"]=4
        with self.assertRaises(ValueError):specification(r)

    def test_feedback_routes_tensor_rules_for_candidate_failures(self):
        r=request(reduction((2,2),[-1]))
        for stage in ("compiler","numerical_comparison","artifact_gate"):
            result=attach(dict(status="failed",failure=dict(layer=stage,diagnostic="PRIVATE_DIAG")),
                          candidate=candidate(r),request=r)
            self.assertIn("non-reduced coordinate",result["public_feedback"]["diagnostic"])
            self.assertNotIn("PRIVATE_DIAG",str(result["public_feedback"]))
        for stage in ("integrity","pipeline","component"):
            result=attach(dict(status="failed",failure=dict(layer=stage)),candidate=candidate(r),request=r)
            self.assertNotIn("Binding and shape guidance",result["public_feedback"]["diagnostic"])

    def test_malformed_source_gets_no_guidance(self):
        r=request(reduction((2,2),[-1]))
        self.assertEqual(repair_hint(r,"def golden(","PRIVATE"),"")
        self.assertEqual(repair_hint(r,"x"*65537,"PRIVATE"),"")
        self.assertEqual(repair_hint(r,candidate(r)["hecate_source"],stage="environment"),"")

    def test_cli_sandbox_and_default_limits_unchanged(self):
        from run_candidate import parse_args
        from sandbox_policy import MODULES
        from application_policy import prepare,BACKEND
        self.assertEqual(parse_args(["--prepare","--case","x","--unified-guidance","explicit-v12"]).unified_guidance,"explicit-v12")
        self.assertIn("unified_tensor_axis_guidance.py",MODULES)
        r=request(reduction((2,2),[0]));old=request(reduction((2,2),[0]),"explicit-v11")
        self.assertEqual(r["rules"],old["rules"])
        _,c=prepare(r["model"],BACKEND,"numerical",{})
        self.assertEqual(c["guidance"],"explicit-v10")

    def test_adapter_compiler_failure_adds_axis_feedback(self):
        from unified_graph_lowering import lower
        from validation_adapter import ValidationContext,CandidateValidationAdapter
        r=request(reduction((2,2),[-1]))
        with tempfile.TemporaryDirectory() as folder:
            report=dict(attempts=[])
            def native(payload,output,command,log):
                if command[0]=="/hecate-opt":
                    log.write_text("synthetic admission failure");return 1
                (output/"trace-evidence.json").write_text(json.dumps(dict(frontend="mock_for_unit_test")))
                log.write_text("mock tracing");return 0
            ctx=ValidationContext(Path(folder),r,report,{},dict(atol=1e-5,rtol=1e-4),45,
                                  Path(folder)/"keys",True,native,lambda:None)
            raw=json.dumps(candidate(r,lower(r)))
            public=CandidateValidationAdapter(ctx).evaluate(raw,0)
            self.assertEqual(public["layer"],"compiler")
            self.assertIn("non-reduced coordinate",public["diagnostic"])
            self.assertFalse(report["attempts"][0]["executed"])

class CompilerContextTests(unittest.TestCase):
    def test_v1_v2_scale_advisory_is_public_but_not_exact_repeat_evidence(self):
        for version in ("v1","v2"):
            r=dict(failure=dict(layer="compiler"),compiler_diagnosis=dict(
                format="fixed-earth-mul-diagnosis-"+version,
                compiler_accumulated_scale=735,compiler_budget=720,
                log="PRIVATE_TOOL_LOG /private/reference"))
            d=from_result(r)
            self.assertEqual(d["context"],dict(compiler_accumulated_scale=735,compiler_budget=720))
            self.assertIsNone(repeat_key(d))
            self.assertNotIn("PRIVATE",json.dumps(d))
            public=attach(r)["public_feedback"]
            self.assertIn("735",public["diagnostic"])
            self.assertNotIn("PRIVATE",json.dumps(public))

    def test_only_bounded_integral_constraints_cross_feedback_boundary(self):
        for value in (True,False,-1,2**53+1,735.0,float("nan"),"PRIVATE"):
            d=from_result(dict(failure=dict(layer="compiler"),compiler_diagnosis=dict(
                format="fixed-earth-mul-diagnosis-v2",compiler_accumulated_scale=value,
                compiler_budget=720,reference=[1,2])))
            self.assertEqual(d["context"],dict(compiler_budget=720))
        for version in ("unknown",None):
            d=from_result(dict(failure=dict(layer="compiler"),compiler_diagnosis=dict(
                format=version,compiler_accumulated_scale=735,compiler_budget=720)))
            self.assertEqual(d["context"],{})

if __name__=="__main__":unittest.main()
