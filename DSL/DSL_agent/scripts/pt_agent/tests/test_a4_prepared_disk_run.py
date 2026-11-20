import copy
from pathlib import Path
import tempfile
import unittest

from pt_agent.experiments import a4_prepared_disk_run as own
from pt_agent.experiments.r31_lattigo_binding import map_precision_plan


def evidence_fixture():
    source='/fixture/source';plan='/fixture/source/plan.json';program='/fixture/source/program.ptdsl'
    prep=dict(status='prepared',variant=own.binding().variant(32),source_case=source,source_plan=plan,
              input='/fixture/input.bin',expected='/fixture/expected.json')
    current=dict(runtime_plan=plan,gate=dict(source_case=source,program=program,factory='original.factory'),
                 source_metadata=dict(plan=dict(execution_count=17863,bootstrap_count=123)))
    math_result=dict(status='ok',worker=dict(status='ok',source_checked=True,math_passed=True,
        mathematical_equivalence_verified=True,replay_attempts=1,replay_completed=1,op_completed=17863))
    proof=dict(status='strict_math_passed',source_plan=plan,current_program=program,source_case=source,
        input=prep['input'],reference=prep['expected'],factory='original.factory',sample_ids=['train:0'],
        operation_count=17863,ideal_bootstrap_count=123,criterion='1e-12+1e-12*abs(reference)')
    expected=dict(sample_id='train:0',old_expected_values_used=False,indices=list(range(10)))
    original=dict(target={},values=[dict(id='fixture',scale_log2=43)],
                  initialization=[],execution=[],finalization=[])
    mapped,_=map_precision_plan(original,own.binding(),bootstrap_cache=32)
    return prep,current,math_result,proof,expected,original,mapped


class PreparedDiskRunnerTests(unittest.TestCase):
    def test_original_math_records_are_sufficient_without_old_native_result(self):
        report=own.validate_records(*evidence_fixture())
        self.assertFalse(report['frozen_program_modified'])
        self.assertFalse(report['raw_data_modified'])

    def test_foreign_math_or_training_sample_reference_is_rejected(self):
        for index,key,value in ((3,'sample_ids',['train:1']),
                                (3,'current_program','/different/program'),
                                (4,'old_expected_values_used',True)):
            fixture=list(evidence_fixture());fixture[index]=dict(fixture[index],**{key:value})
            with self.assertRaises(ValueError):own.validate_records(*fixture)

    def test_failed_math_replay_and_modified_native_plan_are_rejected(self):
        fixture=list(evidence_fixture());fixture[2]['worker']['math_passed']=False
        with self.assertRaises(ValueError):own.validate_records(*fixture)
        fixture=list(evidence_fixture());fixture[-1]['values'][0]['scale_log2']=43
        with self.assertRaises(ValueError):own.validate_records(*fixture)

    def test_parameter_probe_rejects_crypto_activity_and_parameter_drift(self):
        ref={key:None for key in own.PUBLIC_PARAMETER_FIELDS}
        ref.update(variant='fixture',precision_bits=50,residual_parameters={'n':65536})
        actual=dict(ref,status='ok',mode='parameters',keygen_api_attempts=0,keygen_api_completed=0,
            encryption_attempts=0,bootstrap_attempts=0,final_decryption_attempts=0,plaintext_encoding_attempts=0)
        self.assertTrue(own.parameter_check(actual,ref)['passed'])
        actual['keygen_api_attempts']=1
        with self.assertRaises(ValueError):own.parameter_check(actual,ref)
        actual['keygen_api_attempts']=0;actual['residual_parameters']={'n':32768}
        with self.assertRaises(ValueError):own.parameter_check(actual,ref)

    def test_exclusive_execute_argv_keeps_original_binding_and_quiet_flags(self):
        source=dict(plan='/plan',bundle='/bundle',input='/input',expected='/expected')
        argv=own.native_argv(Path('/client'),source,Path('/case'),'execute')
        for name,value in (('-bootstrap-cache','32'),('-precision-bits','50'),('-c2s-bits','52'),
                           ('-auxiliary-prime-bits','61'),('-mod1-inv-degree','15'),
                           ('-log-message-ratio','2'),('-bootstrap-normalization-log2','3')):
            self.assertEqual(argv[argv.index(name)+1],value)
        self.assertNotIn('-check-input-ssa',argv)
        self.assertNotIn('-trace-key-requests',argv)
        self.assertEqual(argv[argv.index('-prepared-dir')+1],'/case/prepared-store')
        parameter_argv=own.native_argv(Path('/client'),source,Path('/case'),'parameters')
        self.assertNotIn('-prepared-dir',parameter_argv)

    def test_resource_timeout_and_driver_categories(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            self.assertEqual(own.terminal_category(dict(status='ok',processes=[]),out),'ok')
            self.assertEqual(own.terminal_category(dict(status='error',resource_guard={'disk':True},processes=[]),out),'resource_limit')
            self.assertEqual(own.terminal_category(dict(status='error',processes=[{'observation':{'timed_out':True}}]),out),'timeout')
            self.assertEqual(own.terminal_category(dict(status='error',processes=[]),out),'driver_error')


if __name__=='__main__':
    unittest.main()
