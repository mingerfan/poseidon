import copy
import json
import unittest
from stage2_directed_guidance_proposal import propose, native, public, digest

def request(module, name):
    value = dict(schema=1, model={"private_marker":"MODEL_CONTENT_NOT_FOR_GUIDANCE"},
                 construction_exercise=module.spec(name))
    if module is public:
        value["construction_profile"] = "hecate-unified-public-v1"
    value["request_id"] = digest(value)
    return value

class DirectedGuidanceProposalTests(unittest.TestCase):
    def test_all_registered_partitions_have_explanations(self):
        count=0
        for module in (native,public):
            for name in module.SPECS:
                source=request(module,name);before=copy.deepcopy(source)
                out=propose(source);count+=1
                self.assertEqual([x["feature"] for x in out["required_features"]],
                                 source["construction_exercise"]["required_features"])
                self.assertEqual(source,before)
                encoded=json.dumps(out)
                for marker in ("MODEL_CONTENT_NOT_FOR_GUIDANCE","coverage_bias","covered_out","coverage_value","@hc.func"):
                    self.assertNotIn(marker,encoded)
        self.assertEqual(count,209)

    def test_composite_includes_zero_dimensional_child_semantics(self):
        out=propose(request(native,"unified-composite-array-storage"))
        entries={x["feature"]:x for x in out["required_features"]}
        self.assertIn("zero-dimensional",entries["return.zero"]["registered_instruction"])
        self.assertIn("[()]",entries["index.zero"]["registered_instruction"])
        self.assertIn("both public and ciphertext",entries["return.mixed"]["registered_instruction"])
        self.assertEqual(len(entries),5)

    def test_public_receiver_constraints(self):
        cases={"attr-shape":"ciphertext Expr","attr-size":"object array",
               "attr-domain":"public polynomial","call-partition":"public string",
               "call-sorted":"public keys"}
        for name,text in cases.items():
            out=propose(request(public,"unified-public-"+name))
            self.assertIn(text,out["required_features"][0]["meaning"])

    def test_mutated_model_rejected(self):
        value=request(native,"unified-item")
        value["model"]["private_marker"]="changed"
        with self.assertRaisesRegex(ValueError,"identity"):
            propose(value)

    def test_mutated_features_even_with_rehashed_request_rejected(self):
        value=request(native,"unified-composite-array-storage")
        value["construction_exercise"]["required_features"].pop()
        value["request_id"]=digest({k:v for k,v in value.items() if k!="request_id"})
        with self.assertRaisesRegex(ValueError,"Changed directed"):
            propose(value)

    def test_wrong_profile_rejected(self):
        value=request(native,"unified-item")
        value["construction_profile"]="hecate-unified-public-v1"
        value["request_id"]=digest({k:v for k,v in value.items() if k!="request_id"})
        with self.assertRaises(ValueError):
            propose(value)

    def test_counter_explains_associated_operation(self):
        out=propose(request(public,"unified-public-counter-comprehensions"))
        row=out["required_features"][0]
        self.assertEqual(row["counted_operation"]["feature"],"node.ListComp")
        self.assertIn("public list comprehension",row["counted_operation"]["meaning"])

    def test_deterministic_and_explicitly_not_integrated(self):
        value=request(native,"unified-item")
        a=propose(value);b=propose(dict(reversed(list(value.items()))))
        self.assertEqual(a,b)
        self.assertFalse(a["runtime_integration_complete"])
        self.assertFalse(a["golden_program_included"])
        self.assertEqual(a["source_request_id"],value["request_id"])

if __name__ == "__main__":
    unittest.main()
