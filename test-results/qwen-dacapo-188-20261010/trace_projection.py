from pathlib import Path
import sys
import numpy as np
import hecate as hc

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "llm_dsl/Qwen25_2Token_Block_DSL"))
from qwen2token.kernels import qwen_layers as ql

with np.load(ROOT / "artifacts/fixture/weights.npz", allow_pickle=False) as weights:
    weight = weights["model_list.0.attention.q_weight.weight"]
    bias = weights["model_list.0.attention.q_weight.bias"]


@hc.func("c")
def qwen_q_projection(x):
    return [ql.linear(ql.PackedVector(x, 896, 32768), weight, bias).expr]


output = ROOT / "artifacts/q-projection"
output.mkdir()
print(hc.save(str(output), str(output)), flush=True)
