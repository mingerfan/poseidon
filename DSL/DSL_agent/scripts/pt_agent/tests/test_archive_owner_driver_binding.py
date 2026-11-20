"""New archive-owner wiring fixtures; old suites and scientific tools are unused."""
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from pt_agent import paths
from pt_agent.experiments import a5r_resnet_rule63_shared_optimized_compile as a
from pt_agent.experiments import orbit_rule63_compile as o
from pt_agent.experiments import archive_producer_closure as owner
from pt_agent.tests import test_a5r_resnet_rule63_shared_optimized_compile as a_fixture
from pt_agent.tests import test_orbit_rule63_compile as o_fixture

class ArchiveOwnerDriverBinding(unittest.TestCase):
    def test_a5r_explicit_owner_is_verified_and_proof_retained(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(paths,'RESULTS_DIR',Path(tmp)):
            args,_,current=a_fixture.A5rResNet().fixture(Path(tmp),archived=True)
            args.archive_supervisor_case=Path(tmp)/'actual-parent'
            proof=dict(closed_producer_births=1,owning_serial_case=str(args.archive_supervisor_case))
            def source_only(case):
                self.assertEqual(case,args.source_case,'No fabricated own archive resource fallback')
                return 9
            with patch.object(a.supervisor,'admission',return_value=current),\
                    patch.object(a.shared.closed_source,'producer_births',side_effect=source_only),\
                    patch.object(owner,'verify_archive_producer',return_value=proof)as verify,\
                    patch.object(a.shutil,'disk_usage',return_value=SimpleNamespace(free=100*a.supervisor.GIB)):
                got=a.source_gate(args)
                self.assertEqual(got['closed_archive_producer_births'],1)
                self.assertEqual(got['archive_producer_proof'],proof)
                verify.assert_called_once_with(Path(tmp)/a.ARCHIVE_ID,args.archive_supervisor_case,
                    host_pid_namespace=args.host_pid_namespace)
                verify.return_value=dict(closed_producer_births=0)
                with self.assertRaisesRegex(ValueError,'Closed archive producer births'):a.source_gate(args)
            parsed=a.arguments(['--source-case',str(args.source_case),'--out',str(args.out),
                '--paper-runs',str(args.paper_runs),'--host-pid-namespace',os.readlink('/proc/self/ns/pid'),
                '--archive-supervisor-case',str(args.archive_supervisor_case)])
            self.assertEqual(parsed.archive_supervisor_case,args.archive_supervisor_case)
            self.assertEqual(parsed.native_seconds,7200)

    def test_orbit_explicit_owner_is_verified_before_payload_preparation(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(paths,'RESULTS_DIR',Path(tmp)),\
                patch.object(o,'RESNET_DEADLINE',datetime(2099,1,1,tzinfo=timezone.utc)):
            fixture=o_fixture.OrbitCompile()
            args,result=fixture.fixture(Path(tmp),archived=True)
            args.archive_supervisor_case=Path(tmp)/'actual-parent'
            proof=dict(closed_producer_births=2,owning_serial_case=str(args.archive_supervisor_case))
            raw=args.source_case/'work/emit/candidate_trace.mlir';gz=raw.with_name(raw.name+'.gz')
            with patch.object(owner,'verify_archive_producer',return_value=proof)as verify,\
                    patch.object(o.closed,'producer_births',side_effect=AssertionError('No own archive fallback')),\
                    patch.object(o.shutil,'disk_usage',return_value=SimpleNamespace(free=100*o.GIB)):
                got=o.archive_gate(args,args.source_case,result,raw,gz)
                self.assertEqual(got['closed_producer_births'],2)
                self.assertEqual(got['archive_producer_proof'],proof)
                verify.assert_called_once_with(args.source_case.parent/(args.source_case.name+'-raw-npz-resource-archive'),
                    args.archive_supervisor_case,host_pid_namespace=os.readlink('/proc/self/ns/pid'))
                verify.return_value=dict(closed_producer_births=0)
                with self.assertRaisesRegex(ValueError,'All saved archive producer births'):o.archive_gate(args,args.source_case,result,raw,gz)
            parsed=o.arguments(['--model','resnet20','--source-case',str(args.source_case),'--out',str(args.out),
                '--paper-runs',str(args.paper_runs),'--archive-supervisor-case',str(args.archive_supervisor_case)])
            self.assertEqual(parsed.archive_supervisor_case,args.archive_supervisor_case)
            self.assertEqual(parsed.native_seconds,7200)

if __name__=='__main__':unittest.main()
