"""E6r2 uses stub provider calls and injected waits; no network."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from pt_agent.llm import LLMError
from pt_agent.planners.provider_retry import ProviderState,ProviderOutage,RetryingBudgetClient,retryable
from pt_agent.planners.transformer_session import model_calls

class Stub:
    def __init__(self,answers):self.answers=iter(answers);self.attempts=[]
    def chat(self,messages,*,purpose):
        self.attempts.append(copy.deepcopy((messages,purpose)));result=next(self.answers)
        if isinstance(result,Exception):raise result
        return result

def error(code='http_error',status=500,size=157):return LLMError(code,dict(status=status,received_bytes=size))
def client(root,answers,*,label='E6r2-a',model='bert',version='t1',limit=8,waits=None):
    stub=Stub(answers)
    return RetryingBudgetClient(stub,root/'calls',provider_state=root/'provider.json',
        model=model,prompt_version=version,label=label,limit=limit,
        sleep=waits.append if waits is not None else lambda _:None),stub

class RetryTests(unittest.TestCase):
    def test_500_success_counts_attempts_with_one_unchanged_request(self):
        with TemporaryDirectory() as t:
            root=Path(t);waits=[];c,s=client(root,[error(),dict(content='ok')],waits=waits)
            messages=[dict(role='user',content='same request')]
            self.assertEqual(c.chat(messages,purpose='p')['content'],'ok')
            self.assertEqual((c.calls,model_calls(root/'calls','bert','t1','E6r2-a')),(2,2))
            self.assertEqual(s.attempts[0],s.attempts[1]);self.assertEqual(waits,[300])
            requests=list(c.directory.glob('*.request.json'));self.assertEqual(len(requests),1)
            self.assertEqual(json.loads(requests[0].read_text())['messages'],messages)
            self.assertEqual(len(list((c.directory/'attempts').glob('*.error.json'))),1)
            self.assertEqual(ProviderState(root/'provider.json').read()['consecutive_retryable'],0)

    def test_retryable_classification(self):
        for code,status,size in [('http_error',503,157),('http_error',429,157),('invalid_response',200,4),
             ('transport_timeout',None,None),('transport_connection_failed',None,None)]:
            self.assertTrue(retryable(error(code,status,size)))
        for code,status,size in [('http_error',400,157),('http_error',401,157),('invalid_response',200,100),
             ('invalid_response',200,1000),('invalid_response',None,4),('settings_error',None,None)]:
            self.assertFalse(retryable(error(code,status,size)))

    def test_short_body_recovers(self):
        with TemporaryDirectory() as t:
            c,s=client(Path(t),[error('invalid_response',200,4),dict(content='ok')])
            self.assertEqual(c.chat([],purpose='p')['content'],'ok');self.assertEqual(len(s.attempts),2)

    def test_400_length_and_full_unparseable_body_do_not_retry(self):
        for response in (error(status=400),error('invalid_response',200,100),
                         dict(content='bad json',finish_reason='length'),dict(content='bad json',finish_reason='stop')):
            with self.subTest(response=response),TemporaryDirectory() as t:
                c,s=client(Path(t),[response])
                if isinstance(response,Exception):
                    with self.assertRaises(LLMError):c.chat([],purpose='p')
                else:self.assertEqual(c.chat([],purpose='p'),response)
                self.assertEqual(c.calls,1);self.assertEqual(len(s.attempts),1)

    def test_three_errors_trip_and_block_all_later_sessions(self):
        with TemporaryDirectory() as t:
            root=Path(t);waits=[];c,s=client(root,[error()]*4,waits=waits)
            with self.assertRaises(ProviderOutage):c.chat([],purpose='p')
            self.assertEqual((c.calls,len(s.attempts)),(3,3));self.assertEqual(waits,[300,300])
            state=ProviderState(root/'provider.json').read();self.assertTrue(state['provider_outage'])
            self.assertEqual(state['consecutive_retryable'],3)
            self.assertTrue(state['outage_window']['start']);self.assertTrue(state['outage_window']['end'])
            other,stub=client(root,[dict(content='unused')],label='p1',model='qwen',version='r8-v11')
            with self.assertRaises(ProviderOutage):other.chat([],purpose='next')
            self.assertEqual(stub.attempts,[]);self.assertEqual(other.calls,0)

    def test_budget_stops_retry_and_streak_survives_new_session(self):
        with TemporaryDirectory() as t:
            root=Path(t);waits=[];c,s=client(root,[error()]*3,limit=2,waits=waits)
            with self.assertRaises(LLMError):c.chat([],purpose='p')
            self.assertEqual((c.calls,len(s.attempts)),(2,2));self.assertEqual(waits,[300])
            with self.assertRaises(LLMError):c.chat([],purpose='exhausted')
            self.assertEqual(len(s.attempts),2)
            other,stub=client(root,[error()],model='qwen')
            with self.assertRaises(ProviderOutage):other.chat([],purpose='next')
            self.assertEqual(len(stub.attempts),1)

    def test_nonretryable_resets_streak_and_cooldown_is_recorded(self):
        with TemporaryDirectory() as t:
            root=Path(t);a,_=client(root,[error()],limit=1)
            with self.assertRaises(LLMError):a.chat([],purpose='p')
            b,_=client(root,[error(status=400)],label='p1',version='r8-v11')
            with self.assertRaises(LLMError):b.chat([],purpose='q')
            waits=[];policy=ProviderState(root/'provider.json',sleep=waits.append)
            self.assertEqual(policy.read()['consecutive_retryable'],0)
            policy.wait('cooldown',600,session='p1')
            self.assertEqual(waits,[600]);self.assertEqual(policy.read()['waits'][-1]['seconds'],600)

    def test_attempts_persist_across_client_processes(self):
        with TemporaryDirectory() as t:
            root=Path(t);c,_=client(root,[error(),dict(content='ok')]);c.chat([],purpose='p')
            other,_=client(root,[dict(content='next')]);self.assertEqual(other.calls,2)
            other.chat([],purpose='q');self.assertEqual(other.calls,3)
            self.assertEqual(len(list(other.directory.glob('*.request.json'))),2)
            self.assertEqual(len(list((other.directory/'attempts').glob('*.error.json'))),1)

if __name__=='__main__':unittest.main()
