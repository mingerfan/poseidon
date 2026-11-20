"""Own small resource metadata/fake-stage checks; original arrays/tools unused."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_vgg_host_cpu_overlap_rss12 as entry
from pt_agent.tests import test_heir_vgg_host_cpu_overlap_tune4 as fixtures
from pt_agent.tests.test_heir_vgg_host_cpu_overlap import put,clean


class RSS12Checks(unittest.TestCase):
    def setUp(self):
        fixtures.Tune4Checks.setUp(self);self.args.out=self.root/(entry.OUT_PREFIX+'fixture')

    def guard(self):
        self.args.out.mkdir(exist_ok=True);return entry._namespace(self.args)['physical_observation']

    def fake(self,*,terminal_rss=False):
        from pt_agent.experiments import heir_common_model_pipeline as pipeline
        def process(command,**options):
            self.calls.append((command,options));put(Path(options['log']),'synthetic process response only\n')
            put(Path(options['resources']),'')
            if '--worker'in command:
                out=Path(command[command.index('--out')+1]);put(out/'result.json',dict(status='ok',lowered_adapter_contract=dict(role='fixture')))
                put(out/'model.mlir','synthetic source metadata only\n')
            return clean()
        def native(*args):
            if terminal_rss:self.table[os.getpid()]['vmrss_kib']=13*1024**2
            return dict(static_bootstrap_operations=2)
        def snapshot(out):(out/'source').mkdir();return[]
        with(patch.object(pipeline,'sources',return_value=self.source),
                patch.object(entry.host.original,'source_gate',return_value=self.checked),
                patch.object(entry.host.original,'_snapshot',side_effect=snapshot),
                patch.object(entry.host.original,'run_process',side_effect=process),
                patch.object(entry.host.original,'check_worker',return_value=dict(passed=True)),
                patch.object(entry.host.original,'check_native',side_effect=native)):
            return entry.execute(self.args)

    def test_aggregate_RSS_above10_up_to12_accepted_and_limits_named12(self):
        guard=self.guard()
        for rss in (10.5,11,11.9):
            with self.subTest(rss=rss):
                row=guard(dict(subtree_rss_bytes=int(rss*entry.GIB)),4*entry.GIB,50*entry.GIB)
                self.assertFalse(row['stop'],row['reasons']);self.assertEqual(row['own_aggregate_RSS_gib'],12)
        self.table[os.getpid()]['vmrss_kib']=0
        self.assertFalse(guard(dict(subtree_rss_bytes=12*entry.GIB),4*entry.GIB,50*entry.GIB)['stop'])

    def test_aggregate_over12_rejected_with_exact12_reason(self):
        row=self.guard()(dict(subtree_rss_bytes=int(12.01*entry.GIB)),4*entry.GIB,50*entry.GIB)
        self.assertTrue(row['stop']);self.assertIn('Own supervisor and descendant RSS exceeds12GiB',row['reasons'])
        self.assertNotIn('Own supervisor and descendant RSS exceeds10GiB',row['reasons'])

    def test_hostavailable2_free10_and_exact_live_identity_still_required(self):
        guard=self.guard()
        self.assertTrue(guard(dict(subtree_rss_bytes=11*entry.GIB),int(1.99*entry.GIB),50*entry.GIB)['stop'])
        self.assertTrue(guard(dict(subtree_rss_bytes=11*entry.GIB),4*entry.GIB,9*entry.GIB)['stop'])
        self.table[93984]['starttime']+=1
        self.assertTrue(guard(None,4*entry.GIB,50*entry.GIB)['stop'])

    def test_worker_native_AS24_source_math_and_RSS12_result_registry_scope(self):
        result=self.fake();self.assertEqual(result['status'],'ok',result['error'])
        self.assertEqual(len(self.calls),2);self.assertTrue(result['model_compiled_accepted'])
        self.assertEqual(result['limits']['AS_gib'],24);self.assertEqual(result['limits']['subtree_RSS_gib'],12)
        self.assertEqual(result['limits']['minimum_available_gib'],2);self.assertEqual(result['limits']['total_seconds'],9600)
        self.assertTrue(result['prior_RSS10_failure_preserved']);self.assertIn('RSS12',result['resource_variant'])
        self.assertEqual([opt['mem_gib']for _,opt in self.calls],[24,24])
        row=json.loads(self.args.registry.read_text());self.assertEqual(len(row),8);self.assertEqual(row['round'],entry.ROUND)
        self.assertIn('RSS allowance is12GiB',row['metrics']['scope'])

    def test_terminal_RSS12_rejection_revokes_acceptance_retaining_native_fact(self):
        # The new native fake finishes, then the original terminal guard sees a
        # current own supervisor footprint above12. No real allocation occurs.
        result=self.fake(terminal_rss=True)
        self.assertNotEqual(result['status'],'ok');self.assertTrue(result['model_compiled'])
        self.assertFalse(result['model_compiled_accepted']);self.assertEqual(result['completed']['native_compile'],1)
        self.assertTrue(result['terminal_parallel_guard']['stop']);self.assertTrue((self.args.out/'completion.json').is_file())

    def test_original_numeric_constant_parser_profile_and_initializer_context_unchanged(self):
        namespace=entry._namespace(self.args)
        for name in('worker','_runtime','run_probes','check_worker','native_command','check_native','serial_lock'):
            self.assertIs(namespace[name],getattr(entry.host.original,name))
        self.assertEqual(entry.host.base.RSS_GIB,10)


if __name__=='__main__':unittest.main()
