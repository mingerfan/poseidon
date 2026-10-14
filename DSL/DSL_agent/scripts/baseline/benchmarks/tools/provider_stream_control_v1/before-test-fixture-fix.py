"""Transport/resource and application controls; all provider calls are fixtures."""
import json,copy,time,tempfile,unittest
from unittest.mock import patch
import deepseek_http_worker as worker
from deepseek_provider import DeepSeekProvider,ProviderError,retryable_failure
from application_diagnostics import make,checked,provider_terminal,render
from application_component import Application
from test_application_component import Backend
from benchmark_suite import Builder

class Lines:
    def __init__(self,lines):self.lines=list(lines);self.bounds=[]
    def readline(self,n):
        self.bounds.append(n)
        if not self.lines:return b""
        line=self.lines.pop(0)
        assert len(line)<=n
        return line

def metrics(info,status="failed",error="transport_invalid_stream"):
    return dict(provider="deepseek_api",calls=[dict(status=status,error=error,transport_diagnostics=info)])

class StreamControlTests(unittest.TestCase):
    def info(self,kind="response_bytes"):
        limit=worker.MAX_RESPONSE_BYTES if kind=="response_bytes"else worker.MAX_SSE_LINE_BYTES
        return dict(stage="body",status=200,stream_error="stream_limit",stream_limit_kind=kind,
                    stream_limit_bytes=limit,stream_observed_bytes=limit+17,received_bytes=limit+17)
    def test_total_byte_limit_records_overshoot_without_accepting_response(self):
        diag={}
        with patch.object(worker,"MAX_RESPONSE_BYTES",128),patch.object(worker,"MAX_SSE_LINE_BYTES",64):
            lines=Lines([b":"+(b"a"*49)+b"\n"]*3)
            with self.assertRaisesRegex(worker.StreamError,"stream_limit"):
                worker.collect_sse(lines,diag,time.monotonic())
            safe=worker.safe_diagnostics(dict(diag,stream_error="stream_limit"))
            self.assertEqual((safe["stream_limit_kind"],safe["stream_limit_bytes"],safe["stream_observed_bytes"]),("response_bytes",128,153))
            self.assertEqual(safe["received_bytes"],153)
            self.assertEqual(lines.bounds,[65]*3)
    def test_line_limit_is_separate(self):
        diag={}
        with patch.object(worker,"MAX_RESPONSE_BYTES",128),patch.object(worker,"MAX_SSE_LINE_BYTES",64):
            with self.assertRaisesRegex(worker.StreamError,"stream_limit"):
                worker.collect_sse(Lines([b":"+(b"a"*64)]),diag,time.monotonic())
            safe=worker.safe_diagnostics(dict(diag,stream_error="stream_limit"))
            self.assertEqual((safe["stream_limit_kind"],safe["stream_limit_bytes"],safe["stream_observed_bytes"]),("line_bytes",64,65))
    def test_done_and_content_assembly_unchanged(self):
        payload=json.dumps({"choices":[{"index":0,"delta":{"content":"ok"}}]}).encode()
        diag={};raw=worker.collect_sse(Lines([b"data: "+payload+b"\n",b"\n",b"data: [DONE]\n",b"\n"]),diag,time.monotonic())
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"],"ok")
        self.assertNotIn("stream_limit_kind",diag)
    def test_missing_done_is_not_resource_limit(self):
        with self.assertRaisesRegex(worker.StreamError,"missing_done"):
            worker.collect_sse(Lines([]),{},time.monotonic())
        self.assertTrue(retryable_failure("transport_invalid_stream",{"stream_error":"missing_done"}))
    def test_old_receipt_still_identifies_limit_without_inventing_kind(self):
        d=provider_terminal(metrics(dict(stage="body",status=200,stream_error="stream_limit")))
        self.assertEqual(d["reason_code"],"provider_stream_limit");self.assertEqual(d["context"],{})
        self.assertEqual(checked(d),d)
    def test_full_numeric_context_reaches_public_application_result(self):
        info=self.info();d=provider_terminal(metrics(info))
        self.assertEqual(d["code"],"provider_response_stream_limit")
        self.assertEqual(d["context"],dict(limit_kind="response_bytes",limit_bytes=worker.MAX_RESPONSE_BYTES,observed_bytes=worker.MAX_RESPONSE_BYTES+17))
        self.assertEqual(d["owner"],"host");self.assertEqual(d["retry_policy"],"review")
        self.assertFalse(d["repeat_key_available"]);self.assertEqual(checked(d),d)
        self.assertIn("response_bytes",render(d)["diagnostic"])
    def test_line_context(self):
        self.assertEqual(provider_terminal(metrics(self.info("line_bytes")))["context"]["limit_kind"],"line_bytes")
    def test_sanitizer_rejects_bad_limit_fields_and_private_strings(self):
        for update in ({"stream_limit_kind":"PRIVATE"},{"stream_limit_bytes":1},{"stream_limit_bytes":True},
                       {"stream_observed_bytes":worker.MAX_RESPONSE_BYTES},
                       {"stream_observed_bytes":worker.MAX_RESPONSE_BYTES+worker.MAX_SSE_LINE_BYTES+2}):
            info=dict(self.info(),**update);safe=worker.safe_diagnostics(dict(info,raw="PRIVATE",headers="PRIVATE"))
            self.assertNotIn("stream_limit_kind",safe)
            self.assertNotIn("PRIVATE",json.dumps(provider_terminal(metrics(info))))
    def test_integer_and_bound_sanitization(self):
        for count in(True,-1,worker.MAX_RESPONSE_BYTES+worker.MAX_SSE_LINE_BYTES+2,"PRIVATE"):
            self.assertNotIn("received_bytes",worker.safe_diagnostics(dict(received_bytes=count)))
        self.assertEqual(worker.safe_diagnostics(dict(received_bytes=worker.MAX_RESPONSE_BYTES+3))["received_bytes"],worker.MAX_RESPONSE_BYTES+3)
    def test_unconfirmed_or_other_transport_error_keeps_generic_failure(self):
        for m in(None,{},dict(provider="fixture",calls=metrics(self.info())["calls"]),
                 metrics(self.info(),status="in_flight"),metrics(self.info(),error="transport_socket_timeout"),
                 metrics({"stream_error":"missing_done"})):
            self.assertEqual(provider_terminal(m),make("provider_failure"))
    def test_resource_limit_remains_nonretryable(self):
        self.assertFalse(retryable_failure("transport_invalid_stream",self.info()))
    def test_invalid_diagnostic_witness_rejected(self):
        for ctx in(dict(limit_kind="PRIVATE",limit_bytes=1,observed_bytes=2),
                   dict(limit_kind="response_bytes",limit_bytes=True,observed_bytes=2),
                   dict(limit_kind="line_bytes",limit_bytes=worker.MAX_SSE_LINE_BYTES,observed_bytes=1)):
            with self.assertRaises(ValueError):make("provider_stream_limit",context=ctx)
        with self.assertRaises(ValueError):make("provider_stream_limit",location={"kind":"candidate"})
    def test_application_returns_specific_terminal_without_retries_or_compilation(self):
        info=self.info()
        class Fixture:
            kind="fixture"
            last_diagnostics={}
            def post(self,body,timeout):
                self.last_diagnostics=copy.deepcopy(info)
                raise ProviderError("transport_invalid_stream")
        with tempfile.TemporaryDirectory()as tmp:
            backend=Backend();app=Application(tmp,backend)
            b=Builder([(2,)]);model=b.finish(b.node("square",["input0"]))
            task=app.submit(model)
            provider=DeepSeekProvider(transport=Fixture())
            result=app.run(task,provider)
            self.assertEqual(result["status"],"service_error")
            self.assertEqual(result["failure"]["code"],"provider_response_stream_limit")
            self.assertEqual(backend.calls,0)
            self.assertEqual(provider.metrics()["generation_attempts"],1)
            self.assertEqual(provider.metrics()["request_attempts"],1)
            self.assertFalse(provider.metrics()["calls"][0]["retry_scheduled"])
if __name__=="__main__":unittest.main()
