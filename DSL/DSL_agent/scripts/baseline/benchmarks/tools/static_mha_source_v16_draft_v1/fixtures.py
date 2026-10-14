"""Small explicit native MHA source contexts; no Torch import until construction."""
CASES = (
    dict(id="self_multihead", e=4, h=2, n=1, l=2, s=2, batch_first=True,
         self_attention=True, need=True, average=False),
    dict(id="separate_dimensions", e=4, h=2, n=1, l=2, s=3, batch_first=True,
         kdim=2, vdim=3, need=True, average=True),
    dict(id="boolean_and_padding", e=2, h=1, n=1, l=2, s=3, batch_first=True,
         need=True, average=True, mask="boolean_padding"),
    dict(id="cross_sequence_first_none", e=2, h=1, n=2, l=2, s=3,
         batch_first=False, need=False, average=True, dropout=.25),
)
APPROX = dict(exp=dict(method="taylor", domain=[-.5, .5], degree=6),
              reciprocal=dict(method="goldschmidt", domain=[.5, 8.], iterations=8))

def build(case, index=0):
    import torch
    prior = torch.get_rng_state()
    try:
        torch.manual_seed(21601 + index)
        attention = torch.nn.MultiheadAttention(
            case["e"], case["h"], dropout=case.get("dropout", 0.),
            kdim=case.get("kdim"), vdim=case.get("vdim"),
            batch_first=case["batch_first"], dtype=torch.float64).eval()
        with torch.no_grad():
            for p in attention.parameters():
                p.uniform_(-.2, .2)
        class Source(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = attention
                self.need = case["need"]
                self.average = case["average"]
                attn = padding = None
                if case.get("mask") == "boolean_padding":
                    attn = torch.tensor([[False, True, False], [True, False, False]], dtype=torch.bool)
                    padding = torch.tensor([[False, True, False]], dtype=torch.bool)
                self.register_buffer("attn_mask", attn)
                self.register_buffer("padding_mask", padding)
            def forward(self, q, k, v):
                return self.attention(q, k, v, key_padding_mask=self.padding_mask,
                                      need_weights=self.need, attn_mask=self.attn_mask,
                                      average_attn_weights=self.average)
        class SelfSource(Source):
            def forward(self, q):
                return self.attention(q, q, q, key_padding_mask=self.padding_mask,
                                      need_weights=self.need, attn_mask=self.attn_mask,
                                      average_attn_weights=self.average)
        module = (SelfSource if case.get("self_attention") else Source)().eval()
        inputs = []
        for role, name in (("query", "q"), ("key", "k"), ("value", "v")):
            if case.get("self_attention") and name != "q":
                continue
            length = case["l"] if role == "query" else case["s"]
            width = case["e"] if role == "query" else case.get("kdim", case["e"]) if role == "key" else case.get("vdim", case["e"])
            dims = [case["n"], length, width] if case["batch_first"] else [length, case["n"], width]
            inputs.append(dict(name=name, shape=dims))
        outputs = ["attn_output", "attn_weights"] if case["need"] else ["attn_output"]
        return module, inputs, outputs
    finally:
        torch.set_rng_state(prior)
