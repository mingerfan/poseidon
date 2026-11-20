"""Allowance expansion preserves the real append-only request ledger."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pt_agent.autonomous_budget import call_records,ensure_allowance,audit_public_messages,RenewingLLMClient
from pt_agent.experiments.r28_common_agent import final_quality_acceptance,image_inputs,agent_options,single_image_input


class AllowanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.ledger=self.root/'calls.jsonl';self.policy=self.root/'allowance.json'

    def test_existing_exhausted_ledger_is_preserved_and_finite_capacity_is_added(self):
        rows=[dict(status=200,seconds=1.0) for _ in range(454)]
        self.ledger.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        a=ensure_allowance(self.policy,self.ledger)
        self.assertEqual(a['ceiling'],574);self.assertEqual(a['observed_attempts'],454)
        self.assertEqual(call_records(self.ledger),rows)
        b=ensure_allowance(self.policy,self.ledger)
        self.assertEqual(b['ceiling'],574);self.assertEqual(len(b['expansions']),1)

    def test_next_request_renews_only_when_existing_capacity_is_consumed(self):
        a=ensure_allowance(self.policy,self.ledger,increment=2)
        self.assertEqual(a['ceiling'],2)
        self.ledger.write_text('{}\n{}\n')
        b=ensure_allowance(self.policy,self.ledger,increment=2)
        self.assertEqual(b['ceiling'],4);self.assertEqual(len(b['expansions']),2)

    def test_changed_ledger_and_malformed_history_are_rejected(self):
        ensure_allowance(self.policy,self.ledger)
        with self.assertRaises(ValueError):ensure_allowance(self.policy,self.root/'other.jsonl')
        self.ledger.write_text('invalid-json\n')
        with self.assertRaises(ValueError):ensure_allowance(self.policy,self.ledger)

    def test_requested_batch_expands_in_whole_increments(self):
        a=ensure_allowance(self.policy,self.ledger,required=250,increment=120)
        self.assertEqual(a['ceiling'],360)
        with self.assertRaises(ValueError):ensure_allowance(self.policy,self.ledger,required=0)

    def test_public_text_contract_excludes_private_paths_and_credentials(self):
        summary=audit_public_messages([dict(role='user',content='Public ResNet-20: conv2d shape=[1,3,32,32]')])
        self.assertEqual(summary['destination'],'https://api.deepseek.com/chat/completions')
        for content in ('read /home/example/private-model','Bearer abcdefghijklmnop','DEEPSEEK_API_KEY=private'):
            with self.subTest(content=content),self.assertRaises(ValueError):
                audit_public_messages([dict(role='user',content=content)])
        with self.assertRaises(ValueError):audit_public_messages([dict(role='user',content=['image'])])

    def test_final_freeze_rejects_legacy_confirmation_with_insufficient_coverage(self):
        row=dict(iteration=1,origin='tuned-r1',passed=True,max_u=0.9412643614780184,
                 acc_drop_pp=-0.7,diverged=False)
        confirmation=dict(selected=dict(iteration=1,origin='tuned-r1'),evals=[row])
        self.assertFalse(final_quality_acceptance(confirmation)['passed'])
        row['max_u']=0.90
        self.assertTrue(final_quality_acceptance(confirmation)['passed'])
        row['acc_drop_pp']=0.500001
        self.assertFalse(final_quality_acceptance(confirmation)['passed'])
        row.update(acc_drop_pp=0,max_u_calibrated=.85,max_u_analytic=1.0001)
        self.assertFalse(final_quality_acceptance(confirmation)['passed'])
        self.assertFalse(final_quality_acceptance(dict(selected=None))['passed'])

    def test_old_feature_role_is_inferred_without_changing_model_input(self):
        original=[dict(name='img',shape=[1,3,32,32],kind='features',dtype='float64')]
        adapted=image_inputs(original)
        self.assertEqual(original[0]['kind'],'features')
        self.assertEqual(adapted[0],dict(original[0],kind='image'))
        with self.assertRaises(ValueError):image_inputs([dict(kind='features',shape=[1,128])])

    def test_current_loop_receives_registered_rule_without_old_loop_options(self):
        options=agent_options(6,600)
        self.assertEqual(options['tune_rule'],'v8-6.25')
        self.assertNotIn('tune_sweeps',options)
        self.assertNotIn('pool_states',options)
        self.assertEqual(options['engine_profile'],'tensor')

    def test_provider_reply_is_saved_before_local_tools_can_fail(self):
        artifacts=self.root/'api'
        reply=dict(content='{"program":"retained source"}',usage=dict(prompt_tokens=3),seconds=.1)
        def respond(messages,*,purpose):
            self.assertEqual(len(list(artifacts.glob('*.request.json'))),1)
            self.assertEqual(len(list(artifacts.glob('*.response.json'))),0)
            return reply
        client=RenewingLLMClient(self.ledger,self.policy,artifact_dir=artifacts)
        with patch('pt_agent.llm.LLMClient',return_value=SimpleNamespace(chat=respond)):
            self.assertEqual(client.chat([dict(role='user',content='public grammar')],purpose='fake'),reply)
        self.assertEqual(json.loads(next(artifacts.glob('*.response.json')).read_text()),reply)

    def test_named_image_container_preserves_original_array_identity(self):
        image=object()
        self.assertIs(single_image_input(dict(img=image),['img']),image)
        self.assertIs(single_image_input(image,['img']),image)
        with self.assertRaises(ValueError):single_image_input(dict(other=image),['img'])


if __name__=='__main__':unittest.main()
