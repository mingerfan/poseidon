"""Renewable application call allowances under the user's autonomous-run authorization."""
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import json
import re
import time
from pathlib import Path


def call_records(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('API call ledger contains a non-object record')
    return rows


def ensure_allowance(policy_path, ledger_path, required=1, increment=120):
    """Raise the recorded absolute ceiling in finite increments; never edit the call ledger."""
    if type(required) is not int or required < 1 or type(increment) is not int or increment < 1:
        raise ValueError('Positive integer call requirements and renewal increments are required')
    policy_path, ledger_path = Path(policy_path), Path(ledger_path)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    with policy_path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        saved = stream.read()
        policy = json.loads(saved) if saved.strip() else dict(
            version=1, authorization='User authorized autonomous budget expansion and all subsequent real API calls on 2026-11-16 Asia/Taipei',
            ledger=str(ledger_path.resolve()), ceiling=0, expansions=[])
        if policy.get('ledger') != str(ledger_path.resolve()) or policy.get('version') != 1:
            raise ValueError('Allowance belongs to a different ledger or policy version')
        used = len(call_records(ledger_path))
        ceiling = policy['ceiling']
        if type(ceiling) is not int or ceiling < 0:
            raise ValueError('Malformed API allowance ceiling')
        if ceiling < used + required:
            new_ceiling = max(used, ceiling) + increment
            while new_ceiling < used + required:
                new_ceiling += increment
            policy['expansions'].append(dict(previous_ceiling=ceiling,new_ceiling=new_ceiling,
                observed_attempts=used,required=required,ts=datetime.now(timezone.utc).isoformat()))
            policy['ceiling'] = new_ceiling
        policy.update(observed_attempts=used, increment=increment)
        stream.seek(0); stream.truncate()
        stream.write(json.dumps(policy, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        stream.flush()
    return policy


def audit_public_messages(messages):
    """Check the text-only public-benchmark request before the original HTTPS worker receives it."""
    if not isinstance(messages,list) or not messages:
        raise ValueError('A nonempty text conversation is required')
    for message in messages:
        if set(message)!={'role','content'} or message['role'] not in ('system','user','assistant') or not isinstance(message['content'],str):
            raise ValueError('Public benchmark requests accept plain role/content messages only')
        text=message['content']
        if re.search(r'/home/|/Users/|[A-Za-z]:\\Users\\|BEGIN [A-Z ]*PRIVATE KEY|Bearer\s+[A-Za-z0-9_-]{12,}|DEEPSEEK_API_KEY\s*=',text):
            raise ValueError('Private paths or credential material are excluded from benchmark prompts')
    return dict(destination='https://api.deepseek.com/chat/completions',message_count=len(messages),
                text_characters=sum(len(m['content']) for m in messages),format='text role/content only',
                raw_images=False,raw_text_samples=False,raw_weight_arrays=False,credentials_in_messages=False)


class RenewingLLMClient:
    """Use the original credential loader and request worker with a renewable recorded ceiling."""
    def __init__(self, budget_path, allowance_path, recorder=None, artifact_dir=None, **settings):
        self.budget_path, self.allowance_path = Path(budget_path), Path(allowance_path)
        self.recorder, self.settings = recorder, settings
        self.artifact_dir=Path(artifact_dir) if artifact_dir is not None else None
        self.calls = 0

    def chat(self, messages, *, purpose):
        from .llm import LLMClient
        audit_public_messages(messages)
        policy = ensure_allowance(self.allowance_path, self.budget_path)
        settings = dict(self.settings, max_total_calls=policy['ceiling'])
        span = self.recorder.span('llm_call', purpose=purpose, absolute_ceiling=policy['ceiling']) if self.recorder else nullcontext()
        with span:
            self.calls += 1
            stem=None
            if self.artifact_dir is not None:
                self.artifact_dir.mkdir(parents=True,exist_ok=True)
                stem=self.artifact_dir/(str(time.time_ns())+'-'+str(self.calls))
                stem.with_suffix('.request.json').write_text(json.dumps(dict(messages=messages,purpose=purpose),ensure_ascii=False)+'\n')
            try:
                reply=LLMClient(self.budget_path, **settings).chat(messages, purpose=purpose)
            except BaseException as exc:
                if stem is not None:stem.with_suffix('.error.json').write_text(json.dumps(dict(type=type(exc).__name__,code=getattr(exc,'code',None)))+'\n')
                raise
            if stem is not None:stem.with_suffix('.response.json').write_text(json.dumps(reply,ensure_ascii=False,allow_nan=False)+'\n')
            return reply
