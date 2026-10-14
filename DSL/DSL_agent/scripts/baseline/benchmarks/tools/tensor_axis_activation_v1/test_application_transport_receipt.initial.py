"""Offline transport receipt and trusted controller checks; no paid requests."""
import copy,io,json,tempfile,time,unittest
from pathlib import Path
from application_transport_receipt import snapshot,MAX_BYTES
from benchmark_graph import canonical,digest
from deepseek_provider import DeepSeekProvider,retryable_failure
from deepseek_http_worker import collect_sse,StreamError
from application_component import Application,Controller
from application_policy import DEFAULT_BUDGET
from benchmark_suite import Builder
from generation_receipt import safe_metrics

def metrics():
    class NoNetwork:
        is_live=True
        def post(self,*a,**k):raise AssertionError("network forbidden in test")
    p=DeepSeekProvider(transport=NoNetwork())
    p.request_attempts=2;p.generation_attempts=1
    p._calls=[
        dict(index=0,generation_index=0,retry_index=0,status="failed",
             error="transport_invalid_stream",retryable=True,retry_scheduled=True,
             retry_delay_seconds=5,
             transport_diagnostics=dict(stage="body",stream_error="missing_done",received_bytes=17)),
        dict(index=1,generation_index=0,retry_index=1,status="failed",
             error="transport_invalid_stream",retryable=False,retry_scheduled=False,
             transport_diagnostics=dict(stage="body",stream_error="invalid_json",sse_events=2))]
    return p.metrics()
class TransportReceiptTests(unittest.TestCase):
    def test_stream_classification_survives_without_changing_retry_decision(self):
        v=metrics();before=copy.deepcopy(v);r=snapshot(v)
        self.assertEqual(v,before)
        self.assertEqual([c["transport_diagnostics"]["stream_error"]for c in r["calls"]],
                         ["missing_done","invalid_json"])
        self.assertTrue(retryable_failure(r["calls"][0]["error"],r["calls"][0]["transport_diagnostics"]))
        self.assertFalse(retryable_failure(r["calls"][1]["error"],r["calls"][1]["transport_diagnostics"]))
        self.assertTrue(r["calls"][0]["retry_scheduled"])
        self.assertFalse(r["calls"][1]["retry_scheduled"])
        self.assertEqual(r["binding"],digest({k:x for k,x in r.items()if k!="binding"}))
        self.assertLessEqual(len(canonical(r)),MAX_BYTES)

    def test_private_body_headers_credentials_and_environment_dropped(self):
        v=metrics();v.update(key="PRIVATE_KEY",proxy="/private/environment",response="PRIVATE_REFERENCE")
        v["calls"][0].update(headers="PRIVATE_HEADERS",body="PRIVATE_BODY",url="PRIVATE_URL")
        v["calls"][0]["transport_diagnostics"].update(
            stage="PRIVATE_STAGE",tls_reason="PRIVATE_CERT",response="PRIVATE_RESPONSE",
            elapsed_seconds=float("nan"),received_bytes=True)
        text=json.dumps(snapshot(v))
        self.assertNotIn("PRIVATE",text)
        self.assertNotIn("elapsed_seconds",text)
        self.assertNotIn("received_bytes",text)

    def test_index_counter_status_and_error_fields_are_allowlisted(self):
        v=metrics();v.update(request_attempts=True,generation_attempts=5,transport_retries=-1)
        v["calls"][0].update(index=True,generation_index=4,retry_index=7,
                            error="PRIVATE_FAILURE",status="PRIVATE_STATUS",retryable="PRIVATE",retry_delay_seconds=999)
        r=snapshot(v)
        for key in ("request_attempts","generation_attempts","transport_retries"):self.assertNotIn(key,r)
        for key in ("index","generation_index","retry_index","error","status","retryable","retry_delay_seconds"):
            self.assertNotIn(key,r["calls"][0])
        self.assertNotIn("PRIVATE",json.dumps(r))

    def test_invalid_or_unbounded_call_ledger_rejected(self):
        for change in (dict(provider="other"),dict(calls=None),dict(calls=[{}]*17),dict(calls=["PRIVATE"])):
            v=metrics();v.update(change)
            with self.assertRaises(ValueError):snapshot(v)

    def test_worker_stream_errors_classified_without_model_call(self):
        for body,expected in ((b"", "missing_done"),(b"data: invalid-json\n\n","invalid_json")):
            diagnostics={}
            with self.assertRaises(StreamError)as error:
                collect_sse(io.BytesIO(body),diagnostics,time.monotonic())
            self.assertEqual(str(error.exception),expected)

    def test_checkpoint_keeps_legacy_metrics_and_adds_separate_receipt(self):
        v=metrics();expected=safe_metrics(v)
        b=Builder([(2,)]);g=b.finish(b.node("square",["input0"]))
        with tempfile.TemporaryDirectory()as root:
            app=Application(Path(root)/"tasks",None)
            task=app.submit(g);path=app._dir(task);state=app._load(path/"state.json")
            c=Controller(app,path,state,DEFAULT_BUDGET)
            c.provider_checkpoint(v)
            self.assertEqual(app._load(path/"provider-metrics.json"),expected)
            receipt=app._load(path/"provider-transport.json")
            self.assertEqual(receipt,snapshot(v))
            self.assertNotIn("transport_diagnostics",json.dumps(expected))
            self.assertEqual(state["ledger"][-1]["outcomes"][0]["error"],"transport_invalid_stream")
            self.assertEqual(state["status"],"queued")

    def test_checkpoint_records_unresolved_attempt_before_failure(self):
        v=metrics();v["calls"]=v["calls"][:1]
        v["calls"][0]=dict(index=0,status="in_flight",generation_index=0,retry_index=0)
        v["request_attempts"]=1
        r=snapshot(v)
        self.assertEqual(r["calls"][0]["status"],"in_flight")
        self.assertEqual(r["calls"][0]["transport_diagnostics"],{})
        self.assertNotIn("retry_scheduled",r["calls"][0])

if __name__=="__main__":unittest.main()
