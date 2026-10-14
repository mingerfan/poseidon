"""NumPy/Torch controls; run only in the existing coordinated Nix Python."""
import copy
import unittest
from fixtures import CASES, APPROX, build
from source_plan import capture, verify, digest, build_request

class MhaMathControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared=[]
        for index,case in enumerate(CASES):
            m,ins,outs=build(case,index)
            p,c=capture(m,ins,outs,approximation=APPROX,trusted=True)
            cls.prepared.append((m,ins,outs,p,c))
    def test_four_contexts_and_v16_only_status(self):
        self.assertEqual(len(self.prepared),4)
        for _,_,_,p,_ in self.prepared:
            verify(p,expected_source_request_sha256=p["source_request_sha256"])
            for field in ("production_intake_integrated","compiler_validated",
                          "encrypted_execution_validated","agent_generated"):
                self.assertIs(p[field],False)
    def test_original_projection_slices_are_explicit(self):
        p=self.prepared[0][3]
        rows=p["normalization"]["calls"][0]["projections"]
        qs=[x for x in rows if x["parameter"]=="in_proj_weight"]
        self.assertEqual([x["slice"] for x in qs],
                         [dict(axis=0,start=0,stop=4),dict(axis=0,start=4,stop=8),dict(axis=0,start=8,stop=12)])
        self.assertEqual(len({x["source_bytes_sha256"] for x in qs}),1)
    def test_separate_dimension_sources_not_packed(self):
        rows=self.prepared[1][3]["normalization"]["calls"][0]["projections"]
        self.assertEqual([x["parameter"] for x in rows if x["role"]!="output" and "weight" in x["parameter"]],
                         ["q_proj_weight","k_proj_weight","v_proj_weight"])
    def test_boolean_mask_direction_and_padding_sources(self):
        row=self.prepared[2][3]["normalization"]["calls"][0]
        self.assertIn("True blocks",row["mask_direction"])
        self.assertEqual(set(row["mask_sources"].values()),{"attn_mask","padding_mask"})
    def test_false_weights_retains_none_without_cipher_slot(self):
        p=self.prepared[3][3]
        self.assertEqual(p["output_record"]["static_none_count"],1)
        self.assertEqual(len(p["mathematical_graph"]["outputs"]),1)
    def test_all_blocked_rows_rejected(self):
        m,ins,outs=build(CASES[2],2)
        m.attn_mask.fill_(True)
        with self.assertRaisesRegex(ValueError,"masked"):
            capture(m,ins,outs,approximation=APPROX,trusted=True)
    def test_float32_and_training_rejected(self):
        m,ins,outs=build(CASES[0])
        for bad in (copy.deepcopy(m).float(),copy.deepcopy(m).train()):
            with self.assertRaises(ValueError):capture(bad,ins,outs,approximation=APPROX,trusted=True)
    def test_negative_infinity_state_is_explicitly_rejected(self):
        import torch
        m,ins,outs=build(CASES[2],2)
        m.attn_mask=torch.tensor([[0.,float("-inf"),0.],[0.,0.,0.]],dtype=torch.float64)
        with self.assertRaisesRegex(ValueError,"finite_public_state"):
            capture(m,ins,outs,approximation=APPROX,trusted=True)
    def test_signed_graph_and_projection_tamper_rejected(self):
        p=copy.deepcopy(self.prepared[0][3])
        p["normalization"]["calls"][0]["projections"][0]["slice"]["start"]=1
        p["binding"]=digest({k:v for k,v in p.items() if k!="binding"})
        with self.assertRaises(ValueError):verify(p,expected_source_request_sha256=p["source_request_sha256"])
    def test_external_source_anchor_rejects_resigned_weight(self):
        p=copy.deepcopy(self.prepared[0][3]);old=p["source_request_sha256"]
        p["source_request"]["source_fx_code"]+="\n# changed"
        p["source_request_sha256"]=digest(p["source_request"])
        p["binding"]=digest({k:v for k,v in p.items() if k!="binding"})
        with self.assertRaisesRegex(ValueError,"external_source_binding"):verify(p,expected_source_request_sha256=old)
    def test_public_state_bytes_cannot_be_rebound_only(self):
        q=copy.deepcopy(self.prepared[0][3]["source_request"])
        first=next(iter(q["public_state"]));q["public_state"][first]["bytes_sha256"]="0"*64
        with self.assertRaises(ValueError):build_request(q)
    def test_dynamic_private_branch_is_not_sample_inferred(self):
        import torch
        attention=copy.deepcopy(self.prepared[0][0].attention)
        class Dynamic(torch.nn.Module):
            def __init__(self):
                super().__init__();self.attention=attention
            def forward(self,q):
                if q.sum()>0:return self.attention(q,q,q)
                return self.attention(-q,-q,-q)
        m=Dynamic().eval();ins=self.prepared[0][1]
        with self.assertRaisesRegex(ValueError,"dynamic_or_unregistered_source"):
            capture(m,ins,["out","weights"],approximation=APPROX,trusted=True)

if __name__=="__main__":
    unittest.main()
