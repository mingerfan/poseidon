"""Poseidon CPU primitive, constant encoding and static-plan tests, no GPU/API."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from hecate_python_env import ROOT, WORK, digest
from run_agent_batch import save


def main():
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from source root")
    os.umask(0o077)
    folder = Path(tempfile.mkdtemp(prefix="poseidon-modswitch-semantics-", dir=WORK / "results"))
    binaries = {
        "primitive": WORK / "build-poseidon/agent-dsl-cpu/bin/test_ckks_modswitch_semantics",
        "level_convention": WORK / "build-poseidon/agent-dsl-adapter/bin/poseidon_mgpu_hevm_plaintext_encoding_tests",
        "static_plan_convention": WORK / "build-poseidon/agent-dsl-adapter/bin/poseidon_mgpu_hevm_static_execution_plan_tests",
    }
    sources = [ROOT / "examples/ckks/test_ckks_modswitch_semantics.cpp",
               ROOT / "src/poseidon/frontends/dacapo/hevm_plaintext_encoding.cpp",
               ROOT / "src/poseidon/frontends/dacapo/hevm_plaintext_encoding.h",
               ROOT / "src/poseidon/tests/frontends/dacapo/hevm_plaintext_encoding_test.cpp",
               ROOT / "src/poseidon/frontends/dacapo/hevm_static_execution_plan.cpp",
               ROOT / "src/poseidon/frontends/dacapo/hevm_static_execution_plan.h",
               ROOT / "src/poseidon/tests/frontends/dacapo/hevm_static_execution_plan_test.cpp",
               Path(__file__).resolve()]
    snapshot = folder / "source"
    snapshot.mkdir()
    hashes = {}
    for index, path in enumerate(sources):
        target = snapshot / (str(index)+"-"+path.name)
        shutil.copyfile(path, target)
        hashes[str(target.relative_to(folder))] = digest(target)
        if digest(path) != hashes[str(target.relative_to(folder))]:
            raise ValueError("Source changed during snapshot")
    report = dict(status="running", gpu_executed=False, hevm_execution=False,
                  agent_calls=0, source_hashes=hashes, checks={})
    print(f"Modswitch semantics evidence: {folder}", flush=True)
    for name, binary in binaries.items():
        try:
            before = digest(binary)
            result = subprocess.run([str(binary)], env=dict(os.environ, OMP_NUM_THREADS="2"),
                                    capture_output=True, timeout=90, check=False)
            (folder / (name+".stdout")).write_bytes(result.stdout)
            (folder / (name+".stderr")).write_bytes(result.stderr)
            if digest(binary) != before:
                raise ValueError("Binary changed during execution")
            item = dict(command=[str(binary)], binary_sha256=before, exit_code=result.returncode,
                        stdout_sha256=digest(folder / (name+".stdout")),
                        stderr_sha256=digest(folder / (name+".stderr")))
            report["checks"][name] = item
            if result.returncode:
                raise ValueError("Native check failed")
            if name == "primitive":
                item["result"] = json.loads(result.stdout)
                if item["result"]["status"] != "passed":
                    raise ValueError("Primitive status mismatch")
            else:
                marker = {"level_convention": b"mgpu HEVM plaintext encoding tests passed",
                          "static_plan_convention": b"mgpu HEVM static execution plan tests passed"}[name]
                if marker not in result.stdout:
                    raise ValueError("Missing native test success")
        except Exception as error:
            report.update(status="failed", failure_layer=name, diagnostic=str(error))
            save(folder / "report.json", report)
            return 1
    report["status"] = "passed"
    save(folder / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
