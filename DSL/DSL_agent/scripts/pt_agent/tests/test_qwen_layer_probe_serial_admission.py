from pathlib import Path
import unittest

from pt_agent.experiments import autonomous_serial_batch as entry


def job():
    return dict(out='/tmp/results/layer', outer_address_space_gib=24, outer_file_gib=64,
        argv=['python3','-B','-m',entry.PREFIX+'qwen_layer_native_probe',
            '--source-case','/tmp/results/frozen','--out','/tmp/results/layer',
            '--memory-gib','24','--case-gib','128','--seconds','10800','--through','compile',
            '--diagnostic-elide-final-ir'])


class LayerProbeAdmission(unittest.TestCase):
    def test_exact_layer_probe_contract_admitted(self):
        source=job()
        self.assertEqual(entry.parent_limits(source),(24,64))
        self.assertEqual(entry.check_command(source['argv'],source['out']),source['argv'])

    def test_resource_changes_or_duplicate_resource_rejected(self):
        for flag,value in [('--memory-gib','32'),('--case-gib','129'),('--seconds','10801')]:
            source=job();source['argv'][source['argv'].index(flag)+1]=value
            with self.assertRaises(ValueError):entry.parent_limits(source)
        source=job();source['argv']+=['--memory-gib','24']
        with self.assertRaises(ValueError):entry.parent_limits(source)

    def test_worker_or_abbreviated_overrides_rejected(self):
        for extra in [['--worker-phase','compile'],['--memory-gib=24'],['--mem','24']]:
            source=job();source['argv']+=extra
            with self.assertRaises(ValueError):entry.parent_limits(source)

    def test_wrong_module_parent_or_case_rejected(self):
        source=job();source['outer_address_space_gib']=32
        with self.assertRaises(ValueError):entry.parent_limits(source)
        source=job();source['argv'][3]=entry.PREFIX+'orbit_common_backend_parse'
        with self.assertRaises(ValueError):entry.parent_limits(source)
        with self.assertRaises(ValueError):entry.check_command(job()['argv'],Path('/tmp/results/other'))


if __name__=='__main__':unittest.main()
