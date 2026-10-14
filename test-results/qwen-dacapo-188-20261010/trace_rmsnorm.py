import json
from pathlib import Path
import sys
import numpy as np
import hecate as hc

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "llm_dsl/Qwen25_2Token_Block_DSL"))
from qwen2token.kernels import qwen_layers as ql

profile = json.loads((ROOT / "llm_dsl/Qwen25_2Token_Block_DSL/config/approximations.json").read_text())
rsqrt = ql.approximation_from_dict(profile["blocks"][0]["pre_norm"])
with np.load(ROOT / "artifacts/fixture/weights.npz", allow_pickle=False) as weights:
    gain = weights["model_list.0.pre_Normal.weight"]


@hc.func("c")
def qwen_rmsnorm(x):
    return [ql.rms_norm(ql.PackedVector(x, 896, 32768), gain, rsqrt=rsqrt).expr]


output = ROOT / "artifacts/rmsnorm"
output.mkdir()
print(hc.save(str(output), str(output)), flush=True)
