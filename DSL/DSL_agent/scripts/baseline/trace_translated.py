"""Trace ONLY trusted deterministic-translator output, never arbitrary Agent code.

No model/reference/key files are needed here. AST checks are defense in depth,
not permission to execute untrusted source; future Agent execution needs OS
capability isolation in addition to these checks.
"""
import argparse
import json
from pathlib import Path
import resource
import numpy as np

from hecate_contract import validate_function
from cipher_abi import physical_input_names
from hecate_python_env import WORK
from seal_artifact_gate import require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    folder = args.directory.resolve()
    require(folder.is_relative_to(WORK / "results"), "Requires native result directory")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (45, 50))
    payload = json.loads((folder / "translation.json").read_text())
    require(payload["generator"] == "deterministic-fx-v0", "Only trusted rule-generated artifacts supported")
    source = (folder / "generated.py").read_text()
    require(source == payload["hecate_source"], "Generated source/manifest mismatch")
    validate_function(source, payload["public_constants"], payload["layout"]["output_ciphertexts"],
                      contract=payload["static_check"]["contract"],
                      input_names=physical_input_names(payload['layout']))
    import hecate as hc
    namespace = {"__builtins__": {}, "hc": hc}
    for name, value in payload["public_constants"].items():
        namespace[name] = np.asarray(value if type(value) is list else [value], dtype=np.float64)
    exec(compile(source, str(folder / "generated.py"), "exec"), namespace)
    hc.save(str(folder), str(folder))


if __name__ == "__main__":
    main()
