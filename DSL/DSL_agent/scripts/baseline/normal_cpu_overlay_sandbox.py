"""Original ordinary-CPU sandbox with one attested read-only DSO directory.

All isolation, limits, commands and other mounts are inherited unchanged. The
selected library is the only content at /selected-poseidon. Loader diagnostics
attest use of that DSO and never expose keys or decrypt intermediate ciphertexts.
"""
import re
import subprocess
from pathlib import Path, PurePosixPath

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha, write_once

TARGET = '/selected-poseidon/libposeidon_shared.so'


def rewrite_command(original, selected_library, *, trace_loader=False):
    cmd = list(original); library = str(Path(selected_library).absolute())
    require(Path(library).name == 'libposeidon_shared.so', 'Selected Poseidon DSO')
    matches = [i for i in range(len(cmd)-2) if cmd[i:i+3] == ['--ro-bind', library, library]]
    require(len(matches) == 1, 'Exactly one original selected-library mount')
    at = matches[0]
    cmd[at:at+3] = ['--dir', '/selected-poseidon', '--ro-bind', library, TARGET]
    entries = [i for i in range(len(cmd)-2) if cmd[i:i+2] == ['--setenv', 'LD_LIBRARY_PATH']]
    require(len(entries) == 1, 'Original library search environment')
    at = entries[0]+2; previous = cmd[at]
    require(previous and all(Path(p).is_relative_to('/nix/store') for p in previous.split(':')), 'Original Nix-only dependencies')
    cmd[at] = '/selected-poseidon:'+previous
    if trace_loader:
        require('LD_DEBUG' not in cmd, 'No inherited loader diagnostics')
        at = cmd.index('--chdir')
        cmd[at:at] = ['--setenv', 'LD_DEBUG', 'libs']
    return cmd


def loaded_library(log, selected_library):
    raw = read_bytes(log, 16*1024**2).decode('utf-8', errors='strict')
    loaded = re.findall(r'^\s*([0-9]+):\s+calling init:\s+(\S*libposeidon_shared\.so)\s*$', raw, re.MULTILINE)
    require(len(loaded) == 1 and loaded[0][1] == TARGET, 'Actual client did not load the selected repaired DSO')
    return bound(dict(format='poseidon-ordinary-CPU-actual-selected-DSO-v1',
        dynamic_loader_pid=int(loaded[0][0]), sandbox_library_path=TARGET,
        selected_library_path=str(Path(selected_library).absolute()), selected_library_sha256=sha(read_bytes(selected_library, 128*1024**2)),
        loader_log_sha256=sha(read_bytes(log, 16*1024**2)), reference_or_parameters_changed=False,
        secret_key_contents_or_hashes_read=False))



def execution_receipt_path(output, argv):
    require(type(argv) is list and len(argv)==6 and argv[0]=='/poseidon-agent-cpu','Original native client arguments')
    destination=PurePosixPath(argv[5])
    require(destination.is_absolute() and '..'not in destination.parts and destination.is_relative_to('/out')
            and destination.suffix=='.json','Native output stays inside the original output mount')
    root=Path(output).resolve();parent=Path(output)/str(destination.relative_to('/out').parent)
    require(parent.is_dir() and parent.resolve()==parent and parent.is_relative_to(root),'Direct native receipt directory')
    return parent/'normal-cpu-overlay-dso.json'


def verify_execution_libraries(output, log, selected_library):
    from poseidon_input_groups import audit, read
    output=Path(output);grouped=audit(output)
    if grouped is None:
        actual=loaded_library(log,selected_library)
        require(actual==checked_bound(read(output/'normal-cpu-overlay-dso.json')),'Actual original selected DSO proof')
        return actual
    clients=[]
    for row in grouped['groups']:
        directory=output/row['directory'];actual=loaded_library(directory/'execute.log',selected_library)
        require(actual==checked_bound(read(directory/'normal-cpu-overlay-dso.json')),'Actual selected DSO proof for every native input group')
        # Each unchanged bwrap call has its own PID namespace; local PIDs may repeat.
        clients.append(dict(index=row['index'],start=row['start'],stop=row['stop'],receipt=actual))
    return bound(dict(format='poseidon-ordinary-CPU-grouped-selected-DSO-v1',input_groups=grouped['inputs'],processes=len(clients),clients=clients,loader_pid_scope='one sandbox PID namespace per invocation',reference_or_parameters_changed=False))


def command(payload, output, argv, *, selected_library, helpers=False, trace_loader=False):
    import poseidon_candidate_sandbox as original
    return rewrite_command(original.command(payload, output, argv, helpers=helpers), selected_library,
                           trace_loader=trace_loader)


def run(payload, output, argv, log, seconds=60, keys=None, *, selected_library, helpers=False):
    require(keys is None, 'Only the unchanged ordinary client ephemeral-key lifecycle')
    from process_resource_limits import limits
    trace = bool(argv) and argv[0] == '/poseidon-agent-cpu'
    receipt_path=execution_receipt_path(output,argv) if trace else None
    selected_hash = sha(read_bytes(selected_library, 128*1024**2))
    from poseidon_sdk_manifest import reject_known_unsafe_key_generator
    reject_known_unsafe_key_generator(dict(artifacts=dict(poseidon_library=selected_hash)))
    cmd = command(payload, output, argv, selected_library=selected_library, helpers=helpers, trace_loader=trace)
    with Path(log).open('x') as stream:
        result = subprocess.run(['/usr/bin/timeout', '-k', '3s', str(seconds), *cmd],
            env={'PATH':'/usr/bin:/bin', 'POSEIDON_PROBE_SECRET':'must-not-survive'},
            stdout=stream, stderr=subprocess.STDOUT, timeout=seconds+10, close_fds=True, preexec_fn=limits)
    if trace and result.returncode == 0:
        receipt = loaded_library(log, selected_library)
        write_once(receipt_path, canonical(receipt))
    return result.returncode
