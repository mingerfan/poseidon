"""Run the unchanged full model computation with progress instrumentation."""
import json
from pathlib import Path
import resource
import runpy
import sys
import time


ROOT = Path("/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1")
PACKAGE = ROOT / "llm_dsl/Qwen25_24Layer_DSL"
OUT = ROOT / "artifacts/qwen24-native"
sys.path.insert(0, str(PACKAGE))
from qwen24.kernels import qwen_model as qm
from qwen24.kernels import qwen_layers as ql

started = time.monotonic()
blocks = []
original_block = qm.Qwen25Model._decoder_block
original_head = ql.lm_head


def event(kind, **fields):
    record = dict(event=kind, elapsed_seconds=time.monotonic() - started,
                  max_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  **fields)
    print(json.dumps(record), flush=True)
    (OUT / "trace-progress.json").write_text(json.dumps(record, indent=2))


def block(model, hidden, positions, cache, index):
    event("block_started", layer=index)
    output = original_block(model, hidden, positions, cache, index)
    blocks.append(index)
    event("block_finished", layer=index)
    return output


def head(*args, **kwargs):
    event("lm_head_started", vocabulary=151936, chunk_size=1024)
    result = original_head(*args, **kwargs)
    event("lm_head_finished", logit_chunks=len(result))
    return result


qm.Qwen25Model._decoder_block = block
ql.lm_head = head
sys.argv = [str(PACKAGE / "trace_qwen24.py"), "--mode", "prefill", "--positions", "0", "1",
            "--weights", str(OUT / "fixture/weights"), "--output-dir", str(OUT / "traced"),
            "--bytecode"]
event("model_loading")
runpy.run_path(sys.argv[0], run_name="__main__")
assert blocks == list(range(24)), blocks
signature = json.loads((OUT / "traced/signature.json").read_text())
assert len(signature["inputs"]) == 2 and len(signature["outputs"]) == 245
event("native_trace_finished", blocks=blocks, outputs=len(signature["outputs"]))
