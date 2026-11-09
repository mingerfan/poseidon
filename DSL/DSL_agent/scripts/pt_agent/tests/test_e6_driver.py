"""Outer E6 orchestration: saved files and stubbed processes only."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pt_agent.experiments import e6_llm_select_tuning as entry

class DriverTests(unittest.TestCase):
    def test_replay_uses_saved_responses_and_tuning_only(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);old=root/'old';run=root/'run';calls=old/'calls/bert/r8-v11';calls.mkdir(parents=True)
            for i in range(1,8):entry.write(calls/f'call-{i:03d}.response.json',dict(content='{}'))
            seen=[]
            def step(command,out,**kwargs):
                phase=command[command.index('--step')+1];session=Path(command[command.index('--root')+1])
                state=entry.read(session/'SESSION.json') if (session/'SESSION.json').exists() else dict(iteration=1)
                i=state['iteration'];seen.append((phase,i,command))
                if phase=='confirm':state['phase']='done';entry.write(session/'AGENT.json',dict(status='confirm_failed'))
                elif phase=='tune':state.update(phase='evaluate',iteration=i+1)
                elif i in (4,5):state['phase']='tune'
                else:state.update(phase='confirm' if i==7 else 'evaluate',iteration=i+1)
                entry.write(session/'SESSION.json',state);return dict(complete=True,exit=0)
            with patch.object(entry,'child',side_effect=step),patch.object(entry,'register'):
                result=entry.replay(run,old,'bert',root/'model',root/'ledger')
            self.assertTrue(result['complete']);self.assertFalse(result['frozen']);self.assertEqual(result['main_api_calls'],0)
            self.assertEqual([i for p,i,c in seen if p=='tune'],[4,5])
            self.assertEqual(sum('--response-file' in c for p,i,c in seen),7)
            for p,i,c in seen:
                self.assertEqual(c[c.index('--tune-call-limit')+1],'12')
                self.assertEqual(c[c.index('--session-label')+1],'E6a')
                self.assertEqual(c[c.index('--session-iters')+1],'7')

    def test_replays_precede_four_ordered_new_sessions(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir();(run/'OFFLINE_TESTS.log').write_text('OK\n')
            entry.write(run/'network-review/manifest.json',dict(order=['registered']));seen=[]
            def replay(*args):seen.append(('E6a',args[2]));return dict(complete=True,frozen=False)
            def process(command,**kwargs):
                model=command[command.index('--model')+1];label=command[command.index('--session-label')+1]
                self.assertEqual(command[command.index('--tune-call-limit')+1],'20');seen.append(('E6b',model,label))
                entry.write(Path(command[command.index('--out')+1])/'REPORT.json',dict(complete=True,frozen=False))
                return SimpleNamespace(returncode=0)
            result=dict(api_before=259,api_after=259,api_delta=0,namespaces=[],accounting_equal=True,within_limits=True)
            with patch.object(entry,'RESULTS_DIR',root),patch.object(entry,'ledger_rows',return_value=[{}]*259), \
                 patch.object(entry,'configurations',return_value={m:dict(model=root/m) for m in ('bert','qwen')}), \
                 patch.object(entry,'replay',side_effect=replay),patch.object(entry,'summarize',return_value=result), \
                 patch.object(entry.subprocess,'run',side_effect=process),patch.object(entry.subprocess,'check_output',return_value=''), \
                 patch.object(entry,'register'):
                self.assertEqual(entry.run(SimpleNamespace(root=run,old=root/'old')),0)
            self.assertEqual(seen,[('E6a','bert'),('E6a','qwen'),('E6b','bert','s1'),('E6b','qwen','s1'),('E6b','bert','s2'),('E6b','qwen','s2')])

class ReportingTests(unittest.TestCase):
    def test_disk_admission_is_a_resource_failure(self):
        report=dict(error='RuntimeError: Trusted runner requires at least 10 GiB free disk space',failure_class='code')
        self.assertEqual(entry.failure_class(1,report),'resource')
        self.assertEqual(entry.failure_class(124,{}),'resource')
        self.assertEqual(entry.failure_class(1,dict(failure_class='provider')),'provider')

    def test_partial_native_retains_both_boot_counts_and_actual_call_tokens(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);session=entry.session_path(root,'e6b','qwen','s1')
            entry.write(session/'AGENT.json',dict(status='feasible',best_source='tuned',best_iteration=3,
                best=dict(boots_pred=22,latency_pred_us=40),prompt_tokens=999,completion_tokens=999))
            entry.write(session/'REPORT.json',dict(seconds=10))
            entry.write(session/'FROZEN.json',{})
            (session/'agent_trace.jsonl').write_text('{"seconds":123}\n')
            for version,count in (('r8-v11',1),('t1',2)):
                for n in range(count):
                    folder=root/'calls/qwen'/version/'s1'
                    entry.write(folder/f'call-{n:03d}.request.json',{})
                    entry.write(folder/f'call-{n:03d}.response.json',dict(seconds=3 if version=='r8-v11' else 2,
                        usage=dict(prompt_tokens=5 if version=='r8-v11' else 4,completion_tokens=2 if version=='r8-v11' else 1)))
            entry.write(session.parent/'acceptance/pars/REPORT.json',dict(complete=False,exit=1,seconds=8,
                export=dict(boots=22),failure_class='code',
                error='RuntimeError: Trusted runner requires at least 10 GiB free disk space'))
            result=entry.session_summary(session)
            self.assertEqual(result['boots'],dict(predicted=22,exported=22,compiled=None))
            self.assertEqual(result['acceptance_status'],'blocked_resource')
            self.assertEqual(result['native_failure_class'],'resource')
            self.assertEqual(result['tokens'],dict(main=7,tuning=10))
            self.assertEqual((result['main_calls'],result['tuning_calls']),(1,2))
            self.assertEqual((result['timings']['main_llm_s'],result['timings']['tuning_llm_s']),(3,4))
            self.assertIsNone(result['timings']['pars_s'])
            self.assertEqual(result['timings']['pars_acceptance_s'],8)

    def test_replay_does_not_bill_historical_llm_seconds_and_compile_clock_is_separate(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);session=entry.session_path(root,'e6a','bert','E6a')
            entry.write(session/'AGENT.json',dict(status='feasible',best_source='llm',best_iteration=1,
                llm_calls=1,best=dict(boots_pred=5),prompt_tokens=999,completion_tokens=999))
            entry.write(session/'FROZEN.json',{})
            (session/'agent_trace.jsonl').write_text('{"seconds":100}\n')
            entry.write(session.parent/'acceptance/pars/REPORT.json',dict(complete=True,seconds=9,
                compile=dict(seconds=2),boots=dict(predicted=5,exported=5,compiled=5)))
            result=entry.session_summary(session)
            self.assertEqual(result['acceptance_status'],'complete')
            self.assertEqual(result['main_calls'],0)
            self.assertEqual(result['tokens']['main'],0)
            self.assertEqual(result['timings']['main_llm_s'],0)
            self.assertEqual(result['timings']['replayed_main_llm_s'],100)
            self.assertEqual(result['timings']['pars_s'],2)
            self.assertEqual(result['timings']['pars_acceptance_s'],9)

class BaselineReportingTests(unittest.TestCase):
    def test_saved_e5d_tune_s_and_bounded_markdown(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);old=root/'old';current=root/'current';current.mkdir()
            (root/'llm_calls.jsonl').write_text('{}\n'*259)
            values={}
            for model in ('bert','qwen'):
                entry.write(old/model/'session/AGENT.json',dict(best=dict(latency_pred_us=12),prompt_tokens=3,completion_tokens=2))
                values[model]=dict(status='feasible',frozen=dict(origin='llm',iteration=1),
                    pars=dict(boots=dict(predicted=5,exported=5,compiled=5),seconds=8),
                    report=dict(engines={}),calls=1,session=dict(seconds=9),
                    timing=dict(llm_s=1,estimate_s=2,validation_s=3,tune_s=7,confirm_s=1,pars_s=2))
            entry.write(old/'E5d.json',dict(models=values))
            with patch.object(entry,'RESULTS_DIR',root):result=entry.summarize(current,old,'disk admission stopped')
            self.assertEqual(result['E5d_comparison']['bert']['timings']['tune_total_s'],7)
            self.assertEqual(result['E5d_comparison']['bert']['timings']['pars_s'],2)
            self.assertEqual(result['E5d_comparison']['bert']['timings']['pars_acceptance_s'],8)
            self.assertLess((current/'E6.md').stat().st_size,10000)
            self.assertEqual(result['e6b']['bert s2']['not_started_reason'],'disk admission stopped')

if __name__=='__main__':unittest.main()
