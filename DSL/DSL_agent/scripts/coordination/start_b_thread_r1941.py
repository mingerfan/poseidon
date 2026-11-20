"""Start the user-requested B conversation through the installed app-server RPC.

Uses normal workspace sandbox and auto_review; never reads project credentials.
Exactly one thread/start and one turn/start. Retains a live stdio connection.
"""
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import time

ROOT = Path('/home/lhohy/Code/Poseidon')
COORD = ROOT / 'DSL/DSL_agent/docs/coordination'
PROMPT = COORD / 'B_THREAD_START_r1941.txt'
PLAN = COORD / 'B_THREAD_LAUNCH_PLAN_r1941.json'
RECEIPT = COORD / 'THREAD_LAUNCH_r1941.json'
PROGRESS = COORD / 'B_THREAD_PROGRESS_r1941.json'
POINTER = COORD / 'THREAD_ROUTING_CURRENT.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal(value):
    return dict(value, binding=digest(value))


def write(path, value, *, new=False):
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode()
    assert len(raw) <= 262144
    if new:
        with path.open('xb') as stream:
            stream.write(raw)
    else:
        temp = path.with_name(path.name + '.r1941.partial')
        temp.write_bytes(raw)
        temp.replace(path)
    return hashlib.sha256(raw).hexdigest()


plan = json.loads(PLAN.read_text())
assert digest({k: v for k, v in plan.items() if k != 'binding'}) == plan['binding']
assert not RECEIPT.exists(), 'Do not create a duplicate B task'
assert hashlib.sha256(PROMPT.read_bytes()).hexdigest() == plan['prompt_sha256']
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['launcher_sha256']
for name, expected in plan['unchanged_sources'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
ledger = json.loads(Path('/home/lhohy/poseidon-work/coordination/v1/ledger.json').read_text())
assert all(x.get('finished') is not None and x.get('active') is None for x in ledger['batches'].values())
state = {'format': 'poseidon-b-thread-launch-v1', 'plan_binding': plan['binding'],
         'projectId': plan['projectId'], 'hostId': plan['hostId'], 'cwd': str(ROOT),
         'B_workdir': '/home/lhohy/Code/Poseidon-resnet20', 'threadId': None,
         'turnId': None, 'created': False, 'started': False,
         'transport': 'installed_codex_app_server_stdio', 'launcher_pid': os.getpid()}
process = subprocess.Popen(['codex', 'app-server', '--listen', 'stdio://'],
                           cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, bufsize=0)
state['app_server_pid'] = process.pid
selector = selectors.DefaultSelector()
selector.register(process.stdout, selectors.EVENT_READ)
buffer = b''
event_counts = {}
last_progress = 0.0
pending_requests = []
ended = False


def send(message):
    process.stdin.write((json.dumps(dict(jsonrpc='2.0', **message)) + '\n').encode())
    process.stdin.flush()


def progress():
    write(PROGRESS, seal(dict(state, event_counts=event_counts,
                             pending_requests=pending_requests,
                             observed_at_unix=time.time())))


def observe(message):
    global last_progress, ended
    method = message.get('method')
    if not method:
        return
    event_counts[method] = event_counts.get(method, 0) + 1
    params = message.get('params', {})
    if 'id' in message:
        # Preserve normal auto_review. Never auto-accept server approval prompts.
        pending_requests.append({'id': message['id'], 'method': method})
        print(json.dumps({'pending_request': method, 'threadId': state['threadId']}), flush=True)
    if method in ('item/started', 'item/completed'):
        item = params.get('item', {})
        state['last_item'] = {'type': item.get('type'), 'id': item.get('id'), 'status': item.get('status')}
    if method == 'turn/started':
        state['active_turn_id'] = params.get('turn', {}).get('id')
        state['turn_status'] = params.get('turn', {}).get('status')
    if method == 'turn/completed':
        state['turn_status'] = params.get('turn', {}).get('status')
        state['last_completed_turn_id'] = params.get('turn', {}).get('id')
        print(json.dumps({'threadId': state['threadId'], 'turn_status': state['turn_status']}), flush=True)
    if time.monotonic() - last_progress >= 2 or method in ('turn/completed', 'error'):
        progress()
        last_progress = time.monotonic()


def receive(timeout):
    global buffer
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if b'\n' in buffer:
            line, buffer = buffer.split(b'\n', 1)
            if line:
                return json.loads(line)
        if selector.select(timeout=min(1, max(0, deadline-time.monotonic()))):
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                raise EOFError('app-server closed')
            buffer += chunk
            assert len(buffer) <= 16 * 1024**2
    return None


def request(number, method, params):
    send({'id': number, 'method': method, 'params': params})
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        message = receive(1)
        if message is None:
            continue
        if message.get('id') == number and 'method' not in message:
            if 'error' in message:
                raise RuntimeError(json.dumps(message['error']))
            return message['result']
        observe(message)
    raise TimeoutError(method)


try:
    request(1, 'initialize', {'clientInfo': {'name': 'poseidon_task_coordinator', 'version': '1.0'},
                              'capabilities': {'experimentalApi': True}})
    send({'method': 'initialized', 'params': {}})
    response = request(2, 'thread/start', plan['thread_start_params'])
    state.update(threadId=response['thread']['id'], created=True,
                 model=response['model'], approvalPolicy=response['approvalPolicy'],
                 approvalsReviewer=response['approvalsReviewer'], sandbox=response['sandbox'])
    write(RECEIPT, seal(state), new=True)
    print(json.dumps({'created_thread': state['threadId'], 'model': state['model'],
                      'approvalPolicy': state['approvalPolicy'], 'approvalsReviewer': state['approvalsReviewer'],
                      'sandbox': state['sandbox']}, ensure_ascii=False), flush=True)
    assert response['cwd'] == str(ROOT)
    assert response['approvalPolicy'] == 'on-request'
    assert response['approvalsReviewer'] == 'auto_review'
    assert response['sandbox']['type'] == 'workspaceWrite'
    assert not response['sandbox'].get('networkAccess', False)
    request(3, 'thread/name/set', {'threadId': state['threadId'], 'name': plan['title']})
    receipt_sha = hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
    write(POINTER, seal({'format': 'poseidon-thread-routing-pointer-v1', 'record': str(RECEIPT),
                         'record_sha256': receipt_sha, 'record_binding': seal(state)['binding'],
                         'B_thread_created': True, 'B_turn_started': False}))
    turn = request(4, 'turn/start', {'threadId': state['threadId'],
                    'input': [{'type': 'text', 'text': PROMPT.read_text()}]})
    state.update(turnId=turn['turn']['id'], started=True, turn_status=turn['turn']['status'])
    receipt_sha = write(RECEIPT, seal(state))
    write(POINTER, seal({'format': 'poseidon-thread-routing-pointer-v1', 'record': str(RECEIPT),
                         'record_sha256': receipt_sha, 'record_binding': seal(state)['binding'],
                         'B_thread_created': True, 'B_turn_started': True}))
    progress()
    print(json.dumps({'started_thread': state['threadId'], 'turnId': state['turnId'],
                      'turn_status': state['turn_status']}, ensure_ascii=False), flush=True)
    # Keep the official server alive; its normal Goal controller owns continuations.
    while True:
        message = receive(1)
        if message is not None:
            observe(message)
        if process.poll() is not None:
            break
except BaseException as error:
    state['launcher_error'] = str(error)[:1000]
    progress()
    print(json.dumps({'launcher_error': str(error)[:1000], 'threadId': state['threadId']}), flush=True)
    raise
finally:
    process.stdin.close()
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
