"""Classify terminal provider failures without inferring billing or DSL ability."""
def provider_failure(report):
 if report.get("status")!="provider_failed":raise ValueError("Terminal provider failure required")
 calls=report.get("provider_metrics",{}).get("calls",[])
 if not calls or calls[-1].get("status")!="failed":raise ValueError("Terminal failed call required")
 error=calls[-1].get("error")
 if not isinstance(error,str):raise ValueError("Missing safe provider failure code")
 if error=="transport_tls_failed":category="tls_failure_unresolved"
 elif error in ("http_status_401","http_status_403"):category="authentication_or_access"
 elif error=="http_status_429":category="provider_rate_limit"
 elif error.startswith("http_status_5"):category="provider_server_failure"
 elif error.startswith("transport_"):category="transport_failure"
 else:category="provider_failure_unclassified"
 return dict(layer="provider",category=category,error_code=error,
  terminal_retryable=calls[-1].get("retryable"),http_attempts=len(calls),
  received_responses=sum(c.get("status")=="response_received" for c in calls),
  attempts_evaluated_before_provider_failure=len(report.get("attempts",[])),
  compiled_before_provider_failure=any(a.get("compiled") for a in report.get("attempts",[])),
  encrypted_before_provider_failure=any(a.get("executed") for a in report.get("attempts",[])),
  automatic_retry_authorized_by_classification=False,billing_status="not_determined",
  limitation="TLS code alone does not distinguish certificate validation, handshake termination or local network cause; no DSL support conclusion")
