"""Only trusted fixed workers are launched by the host; candidates cannot select one."""
import json
import os
from pathlib import Path
import resource
import socket
import sys


def probe():
    from candidate_trace import load_frontend
    load_frontend()  # Fail infrastructure preflight before spending repair attempts.
    payload = json.loads(Path("/payload.json").read_text())
    checks = {
        "frontend_loaded": True,
        "private_sentinel_hidden": not Path(payload["sentinel"]).exists(),
        "workspace_hidden": not Path(payload["workspace"]).exists(),
        "keys_hidden": not Path("/keys").exists(),
        "environment_cleared": "POSEIDON_PROBE_SECRET" not in os.environ,
        "network_namespace_changed": os.readlink("/proc/self/ns/net") != payload["net_ns"],
        "pid_namespace_changed": os.readlink("/proc/self/ns/pid") != payload["pid_ns"],
        "only_loopback_interface": set(name for _, name in socket.if_nameindex()) <= {"lo"},
    }
    try:
        with open("/payload.json", "a"):
            pass
        checks["request_read_only"] = False
    except OSError:
        checks["request_read_only"] = True
    expected_limits = {'AS': (4 * 1024**3, 4 * 1024**3), 'CPU': (150, 155),
                       'FSIZE': (16 * 1024**2, 16 * 1024**2), 'NOFILE': (128, 128),
                       'CORE': (0, 0)}
    measured_limits = {name: resource.getrlimit(getattr(resource, 'RLIMIT_' + name))
                       for name in expected_limits}
    checks['resource_limits_applied'] = measured_limits == expected_limits
    checks['resource_limits'] = measured_limits
    Path("/out/probe.json").write_text(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise RuntimeError("Sandbox capability probe failed")


def main():
    if sys.argv[1:] == ["probe"]:
        probe()
    elif sys.argv[1:] == ["execute"]:
        from seal_cpu_golden import execute_artifact
        from candidate_contract import request_rotations, request_input_names
        from cipher_abi import execution_options, artifact_options
        payload = json.loads(Path("/payload.json").read_text())
        if 'compiler_configuration' in payload['request']:
            from compiler_configuration import verify_artifact_configuration
            from hecate_python_env import digest
            from seal_artifact_gate import inspect_artifacts
            gate = inspect_artifacts(Path('/out/lowered._hecate_golden.hevm').read_bytes(),
                                     Path('/out/_hecate_golden.cst').read_bytes(),
                                     rotation_steps=request_rotations(payload['request']),
                                     expected_inputs=len(request_input_names(payload['request'])),
                                     **artifact_options(payload['request']['layout']))
            verify_artifact_configuration(payload['request'], gate, digest(Path('/profile.json')))
        execute_artifact(Path("/out"), Path("/keys"), payload["request"]["layout"]["output_selectors"],
                         rotation_steps=request_rotations(payload["request"]),
                         **execution_options(payload['request']['layout']))
        layout=payload['request']['layout']
        if 'execution_abi' in layout and len(layout['output_shape'])>1:
            import numpy as np
            from packed_input_abi import decode_output
            logical=decode_output(np.load('/out/decrypted.npy',allow_pickle=False),layout)
            np.save('/out/decrypted-logical.npy',logical,allow_pickle=False)
    else:
        raise ValueError("Unknown trusted worker")


if __name__ == "__main__":
    main()
