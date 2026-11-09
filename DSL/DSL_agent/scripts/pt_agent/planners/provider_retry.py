"""E6r2 provider attempts and a shared, persistent outage circuit.

Only this opt-in client retries. The original ModelBudgetClient is unchanged.
"""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import time

from ..llm import LLMError
from .transformer_session import ModelBudgetClient

RETRY_WAIT_S=300
COOLDOWN_S=600
MAX_ATTEMPTS=3
CONNECTION_ERRORS=frozenset(('transport_timeout','transport_socket_timeout',
    'transport_connection_failed','transport_dns_failed','transport_remote_disconnected',
    'transport_incomplete_read'))


def retryable(error):
    diagnostics=error.diagnostics or {};status=diagnostics.get('status')
    if error.code in CONNECTION_ERRORS:return True
    if error.code=='http_error':return isinstance(status,int) and (500<=status<=599 or status==429)
    size=diagnostics.get('received_bytes')
    return error.code=='invalid_response' and status==200 and isinstance(size,int) and 0<=size<100


def wait_chunks(seconds):
    deadline=time.monotonic()+seconds
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0:return
        time.sleep(min(30,remaining))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


class ProviderOutage(LLMError):
    def __init__(self):
        super().__init__('llm_error');self.code='provider_outage';self.args=('provider_outage',)


class ProviderState:
    def __init__(self,path,*,sleep=wait_chunks,clock=None):
        self.path=Path(path);self.sleep=sleep
        self.clock=clock or (lambda:datetime.now(timezone.utc).isoformat())

    def read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else dict(
            consecutive_retryable=0,provider_outage=False,window_start=None,attempts=[],waits=[])

    def update(self,mutate):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as stream:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
            state=self.read();mutate(state);write(self.path,state);return state

    def require_available(self):
        if self.read()['provider_outage']:raise ProviderOutage()

    def record(self,*,namespace,request,attempt,error=None):
        now=self.clock()
        def change(state):
            can_retry=error is not None and retryable(error)
            state['attempts'].append(dict(ts=now,namespace=namespace,request=request,attempt=attempt,
                error=error.code if error else None,retryable=can_retry))
            if can_retry:
                if state['consecutive_retryable']==0:state['window_start']=now
                state['consecutive_retryable']+=1
                if state['consecutive_retryable']>=3:
                    state['provider_outage']=True
                    state['outage_window']=dict(start=state['window_start'],end=now)
            else:
                state['consecutive_retryable']=0;state['window_start']=None
        return self.update(change)

    def wait(self,kind,seconds,*,namespace=None,request=None,session=None):
        self.require_available()
        event=dict(kind=kind,seconds=seconds,start=self.clock(),namespace=namespace,request=request,session=session)
        self.update(lambda state:state['waits'].append(event))
        self.sleep(seconds)
        end=self.clock()
        self.update(lambda state:state['waits'][-1].update(end=end))


class RetryingBudgetClient(ModelBudgetClient):
    """One logical request file, one reservation/result per provider attempt."""
    def __init__(self,client,directory,*,provider_state,model,prompt_version,label,limit=40,
                 sleep=wait_chunks,retry_wait_s=RETRY_WAIT_S):
        super().__init__(client,directory,model=model,prompt_version=prompt_version,label=label,limit=limit)
        self.policy=ProviderState(provider_state,sleep=sleep)
        self.namespace=[model,prompt_version,label];self.retry_wait_s=retry_wait_s
        marker=self.directory/'provider-policy.json'
        configuration=dict(provider_state=str(provider_state),max_attempts=MAX_ATTEMPTS,retry_wait_s=retry_wait_s)
        if marker.exists():
            if json.loads(marker.read_text())!=configuration:raise ValueError('Provider retry policy changed within a namespace')
        else:write(marker,configuration)

    @property
    def calls(self):return len(list((self.directory/'attempts').glob('*.record.json')))

    def chat(self,messages,*,purpose):
        with (self.directory/'client.lock').open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            self.policy.require_available()
            if self.calls>=self.limit:raise LLMError('budget_exhausted')
            stem='call-'+str(len(list(self.directory.glob('call-*.request.json')))+1).zfill(3)
            request_path=self.directory/(stem+'.request.json')
            write(request_path,dict(messages=messages,purpose=purpose))
            for attempt in range(1,MAX_ATTEMPTS+1):
                self.policy.require_available()
                prefix=self.directory/'attempts'/f'{stem}-{attempt:03d}'
                write(Path(str(prefix)+'.record.json'),dict(namespace=self.namespace,request=str(request_path),
                    purpose=purpose,attempt=attempt,ts=self.policy.clock()))
                started=time.monotonic()
                try:reply=self.client.chat(messages,purpose=purpose)
                except LLMError as error:
                    failure=dict(error=error.code,diagnostics=error.diagnostics,attempt=attempt,purpose=purpose,
                        seconds=(error.diagnostics or {}).get('elapsed_seconds',time.monotonic()-started))
                    write(Path(str(prefix)+'.error.json'),failure)
                    state=self.policy.record(namespace=self.namespace,request=str(request_path),attempt=attempt,error=error)
                    if state['provider_outage']:
                        write(self.directory/(stem+'.error.json'),dict(failure,provider_outage=True))
                        raise ProviderOutage() from None
                    if not retryable(error) or attempt==MAX_ATTEMPTS or self.calls>=self.limit:
                        write(self.directory/(stem+'.error.json'),failure);raise
                    self.policy.wait('retry',self.retry_wait_s,namespace=self.namespace,request=str(request_path))
                else:
                    write(Path(str(prefix)+'.response.json'),reply)
                    self.policy.record(namespace=self.namespace,request=str(request_path),attempt=attempt)
                    write(self.directory/(stem+'.response.json'),reply)
                    return reply
