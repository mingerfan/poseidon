"""Offline protocol regression: no inference, tracing, or changed acceptance rules."""
import copy
import hashlib
import json
import ssl
import unittest

from candidate_contract import (make_request, canonical, TASK_RULES, request_input_names,
                                validate_candidate)
from deepseek_provider import public_request, retryable_failure
from deepseek_http_worker import tls_diagnostics, safe_diagnostics


def fixture(count=2):
    names = ('x', 'y', 'z', 't')[:count]
    layout = dict(inputs=[dict(name='input'+str(i),dsl_name=n,shape=[4]) for i,n in enumerate(names)],
                  output_ciphertexts=1)
    return make_request(dict(static_check=dict(contract='hecate-function-v3'),
        fx_graph='public graph', public_constants={}, constant_origins={}, layout=layout),
        dict(schema=3), 'a'*64)


class MultiInputProtocolTests(unittest.TestCase):
    def test_exact_headers_for_two_three_four_inputs(self):
        for count in (2,3,4):
            request=fixture(count)
            names=request_input_names(request)
            header='@hc.func("' + ','.join(['c']*count) + '")'
            signature='def golden(' + ', '.join(names) + '):'
            self.assertIn(header, request['rules'])
            self.assertIn(signature, request['rules'])
            self.assertEqual(public_request(request),request)
            source=header+'\n'+signature+'\n    return '+' + '.join(names)+'\n'
            result=validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=source),request)
            self.assertEqual(result['contract'],'hecate-function-v3')

    def test_old_v3_requests_remain_accepted_and_hash_distinct(self):
        request=fixture()
        old=copy.deepcopy(request)
        old['task']='hecate-function-synthesis-v3'
        old['rules']=TASK_RULES[old['task']][1]
        old['request_id']=hashlib.sha256(canonical({k:v for k,v in old.items() if k!='request_id'})).hexdigest()
        self.assertEqual(public_request(old),old)
        self.assertNotEqual(old['request_id'],request['request_id'])
        source='@hc.func("c,c")\ndef golden(x, y):\n    return x + y\n'
        for req in (old,request):
            validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)

    def test_observed_malformed_headers_still_rejected_with_actionable_feedback(self):
        request=fixture()
        for source,expected in (
            ('@c(c,c)\ndef other(left, right):\n    return left + right\n','def golden(x, y):'),
            ('def golden(left, right):\n    return left + right\n','def golden(x, y):'),
            ('def golden(x, y):\n    return x + y\n','@hc.func("c,c")'),
            ('@c(c,c)\ndef golden(x, y):\n    return x + y\n','@hc.func("c,c")'),
            ('@hc.func("c", "c")\ndef golden(x, y):\n    return x + y\n','@hc.func("c,c")')):
            with self.assertRaises(ValueError) as caught:
                validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=source),request)
            self.assertIn(expected,str(caught.exception))

    def test_tls_subtypes_are_safe_and_do_not_enable_insecure_retries(self):
        for error,kind in ((ssl.SSLEOFError('SECRET'),'unexpected_eof'),
                           (ssl.SSLCertVerificationError('SECRET'),'certificate_verification'),
                           (ssl.SSLError('SECRET'),'protocol_error')):
            diagnostics=tls_diagnostics(error)
            self.assertEqual(diagnostics['tls_error'],kind)
            self.assertEqual(safe_diagnostics(diagnostics),diagnostics)
            self.assertNotIn('SECRET',json.dumps(diagnostics))
            self.assertFalse(retryable_failure('transport_tls_failed',diagnostics))
        self.assertEqual(safe_diagnostics(dict(tls_error='SECRET',tls_reason='SECRET')), {})


if __name__=='__main__':
    unittest.main()
