"""Private stdlib HTTPS worker; -I with a hard parent timeout. No retries.

No redirects/proxy discovery. An explicit DeepSeek-only loopback CONNECT route
is optional. Default certificate and hostname checks stay on.
Never run manually with a real key; API usage needs separate approval.
"""
import http.client
import json
from pathlib import Path
import socket
import ssl
import sys
import time
import math

MAX_BYTES = 1024**2
MAX_RESPONSE_BYTES = 32 * 1024**2
ROUTES = {
    "deepseek": ("api.deepseek.com", "/chat/completions"),
}


class TrustStoreError(RuntimeError):
    pass


class StreamError(ValueError):
    pass


STAGES = ('setup', 'connect', 'send', 'headers', 'body', 'complete')
ERROR_CODES = ('invalid_request_error', 'rate_limit_error', 'insufficient_quota',
               'authentication_error', 'permission_error', 'model_not_found', 'server_error')
PARAMS = ('model', 'max_tokens', 'max_completion_tokens', 'reasoning_effort',
          'thinking', 'response_format', 'messages', 'stream', 'tools', 'reasoning_content')
HINTS = ('max_tokens', 'reasoning_content', 'response_format', 'rate_limit',
         'quota', 'balance', 'model', 'upstream', 'coding', 'overloaded', 'unknown')
STREAM_ERRORS = ('missing_done', 'stream_limit', 'invalid_event', 'invalid_choices',
                 'invalid_choice_index', 'unexpected_action', 'invalid_delta', 'invalid_json')
TLS_ERRORS = ('certificate_verification', 'unexpected_eof', 'clean_shutdown', 'protocol_error')
TLS_REASONS = ('UNEXPECTED_EOF_WHILE_READING', 'DECRYPTION_FAILED_OR_BAD_RECORD_MAC',
               'SSLV3_ALERT_BAD_RECORD_MAC', 'CERTIFICATE_VERIFY_FAILED',
               'TLSV1_ALERT_INTERNAL_ERROR', 'WRONG_VERSION_NUMBER', 'UNSUPPORTED_PROTOCOL')


def tls_diagnostics(error):
    # Class/allowlisted OpenSSL reason only, never exception text or peer data.
    if not isinstance(error, ssl.SSLError):
        return {}
    kind = ('certificate_verification' if isinstance(error, ssl.SSLCertVerificationError)
            else 'unexpected_eof' if isinstance(error, ssl.SSLEOFError)
            else 'clean_shutdown' if isinstance(error, ssl.SSLZeroReturnError)
            else 'protocol_error')
    result = dict(tls_error=kind)
    reason = getattr(error, 'reason', None)
    if type(reason) is str and reason in TLS_REASONS:
        result['tls_reason'] = reason
    return result


def safe_error(raw):
    """Never return free-form provider text, even if it echoes credentials."""
    result = {'error_code': 'unknown', 'error_param': 'unknown', 'error_hint': 'unknown'}
    try:
        value = json.loads(raw)
        error = value.get('error', value)
        if not isinstance(error, dict):
            return result
        for source in ('code', 'type'):
            if error.get(source) in ERROR_CODES:
                result['error_code'] = error[source]
        if error.get('param') in PARAMS:
            result['error_param'] = error['param']
        message = str(error.get('message', '')).lower()
        for hint in HINTS[:-1]:
            if hint in message:
                result['error_hint'] = hint
                break
    except (ValueError, TypeError, AttributeError):
        pass
    return result


def safe_diagnostics(data):
    """Second allowlist at the parent boundary; reject arbitrary worker strings."""
    if not isinstance(data, dict):
        return {}
    out = {}
    enums = {'stage': STAGES, 'error_code': (*ERROR_CODES, 'unknown'),
             'error_param': (*PARAMS, 'unknown'), 'error_hint': HINTS, 'stream_error': STREAM_ERRORS,
             'tls_error': TLS_ERRORS, 'tls_reason': TLS_REASONS}
    for key, allowed in enums.items():
        if type(data.get(key)) is str and data[key] in allowed:
            out[key] = data[key]
    for key in ('status', 'received_bytes', 'sse_events'):
        if type(data.get(key)) is int and 0 <= data[key] <= MAX_RESPONSE_BYTES + 1:
            out[key] = data[key]
    for key in ('elapsed_seconds', 'connect_seconds', 'send_seconds', 'headers_seconds',
                'body_seconds', 'first_body_byte_seconds'):
        if type(data.get(key)) in (int, float) and math.isfinite(data[key]) and 0 <= data[key] <= 10000:
            out[key] = data[key]
    return out


def collect_sse(response, diagnostics, started):
    """Bounded SSE assembly. No partial result is accepted without [DONE]."""
    result = {'object': 'chat.completion', 'choices': [{'index': 0, 'message':
              {'role': 'assistant', 'content': '', 'reasoning_content': ''}, 'finish_reason': None}]}
    message = result['choices'][0]['message']
    content, reasoning, event = [], [], []
    received = events = 0
    while True:
        line = response.readline(1024 * 1024 + 1)
        if not line:
            raise StreamError('missing_done')
        received += len(line)
        diagnostics['received_bytes'] = received
        diagnostics.setdefault('first_body_byte_seconds', time.monotonic()-started)
        if received > MAX_RESPONSE_BYTES or len(line) > 1024 * 1024:
            raise StreamError('stream_limit')
        if line.strip():
            if line.startswith(b'data:'):
                event.append(line[5:].strip())
            continue
        if not event:
            continue
        payload = b'\n'.join(event)
        event = []
        if payload == b'[DONE]':
            message['content'], message['reasoning_content'] = ''.join(content), ''.join(reasoning)
            return json.dumps(result, allow_nan=False).encode()
        events += 1
        diagnostics['sse_events'] = events
        try:
            obj = json.loads(payload)
        except (ValueError, UnicodeError):
            raise StreamError('invalid_json') from None
        if not isinstance(obj, dict) or 'error' in obj:
            raise StreamError('invalid_event')
        for key in ('id', 'model', 'system_fingerprint', 'usage'):
            if obj.get(key) is not None:
                result[key] = obj[key]
        choices = obj.get('choices', [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise StreamError('invalid_choices')
        for choice in choices:
            if choice.get('index', 0) != 0:
                raise StreamError('invalid_choice_index')
            delta = choice.get('delta', {})
            if delta.get('tool_calls') or delta.get('refusal'):
                raise StreamError('unexpected_action')
            for key, parts in (('content', content), ('reasoning_content', reasoning)):
                if delta.get(key) is not None:
                    if not isinstance(delta[key], str):
                        raise StreamError('invalid_delta')
                    parts.append(delta[key])
            if choice.get('finish_reason') is not None:
                result['choices'][0]['finish_reason'] = choice['finish_reason']


def failure_code(error):
    # Classify by type only: native error messages can contain credentials.
    if isinstance(error, ssl.SSLCertVerificationError):
        return "transport_certificate_failed"
    if isinstance(error, ssl.SSLError):
        return "transport_tls_failed"
    if isinstance(error, socket.gaierror):
        return "transport_dns_failed"
    if isinstance(error, TimeoutError):
        return "transport_socket_timeout"
    if isinstance(error, TrustStoreError):
        return "transport_trust_store_unavailable"
    if isinstance(error, http.client.RemoteDisconnected):
        return 'transport_remote_disconnected'
    if isinstance(error, http.client.IncompleteRead):
        return 'transport_incomplete_read'
    if isinstance(error, http.client.BadStatusLine):
        return 'transport_bad_status_line'
    if isinstance(error, StreamError):
        return 'transport_invalid_stream'
    if isinstance(error, http.client.HTTPException):
        return "transport_http_protocol_failed"
    if isinstance(error, OSError):
        return "transport_connection_failed"
    if isinstance(error, (ValueError, KeyError, TypeError)):
        return "transport_invalid_payload"
    return "transport_worker_failed"


def tls_context():
    context = ssl.create_default_context()
    # Nix Python's compiled default trust path may not be populated. Reuse only
    # the existing OS trust bundle, never a downloaded cert or an insecure mode.
    system_bundle = Path("/etc/ssl/certs/ca-certificates.crt")
    if not context.get_ca_certs() and system_bundle.is_file():
        context.load_verify_locations(cafile=str(system_bundle))
    if not context.get_ca_certs():
        raise TrustStoreError("No installed trusted CA bundle")
    return context


def exchange(wire, diagnostics=None):
    diagnostics = {} if diagnostics is None else diagnostics
    started = time.monotonic()
    diagnostics['stage'] = 'setup'
    if len(wire) > MAX_BYTES:
        raise ValueError("request_limit")
    payload = json.loads(wire)
    body = json.dumps(payload["body"], allow_nan=False, separators=(",", ":")).encode()
    provider = payload.get("provider", "deepseek")
    host, path = ROUTES[provider]
    headers = {"Authorization": "Bearer " + payload["api_key"],
               "Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": "Poseidon-FHE-DSL-Agent/0.1"}
    port = payload.get('loopback_proxy_port', 0)
    if type(port) is not int or not 0 <= port <= 65535 or (port and provider != 'deepseek'):
        raise ValueError('invalid_loopback_proxy_port')
    if port:
        # CONNECT carries no API credentials. HTTPSConnection wraps the tunnel
        # with SNI/hostname validation for the fixed destination, not localhost.
        connection = http.client.HTTPSConnection('127.0.0.1', port=port,
                                                 timeout=payload["timeout"], context=tls_context())
        connection.set_tunnel(host, 443)
    else:
        connection = http.client.HTTPSConnection(host, timeout=payload["timeout"],
                                                 context=tls_context())
    try:
        diagnostics['stage'] = 'connect'
        tick = time.monotonic()
        connection.connect()
        diagnostics['connect_seconds'] = time.monotonic()-tick
        diagnostics['stage'] = 'send'
        tick = time.monotonic()
        connection.request("POST", path, body=body, headers=headers)
        diagnostics['send_seconds'] = time.monotonic()-tick
        diagnostics['stage'] = 'headers'
        tick = time.monotonic()
        response = connection.getresponse()
        diagnostics['headers_seconds'] = time.monotonic()-tick
        diagnostics['status'] = response.status
        diagnostics['stage'] = 'body'
        tick = time.monotonic()
        if response.status == 200:
            if payload['body'].get('stream'):
                data = collect_sse(response, diagnostics, started)
            else:
                data = response.read(MAX_RESPONSE_BYTES + 1)
                diagnostics['received_bytes'] = len(data)
        else:
            # Read only a bounded body; persist finite classifications, never text.
            data = b''
            if callable(getattr(response, 'read', None)):
                if getattr(connection, 'sock', None):
                    connection.sock.settimeout(min(15, payload['timeout']))
                raw = response.read(8193)
                diagnostics['received_bytes'] = len(raw)
                if len(raw) <= 8192:
                    diagnostics.update(safe_error(raw))
        diagnostics['body_seconds'] = time.monotonic()-tick
        if len(data) > MAX_RESPONSE_BYTES:
            raise ValueError("response_limit")
        diagnostics['stage'] = 'complete'
        return str(response.status).encode() + b"\n" + data
    except StreamError as error:
        if str(error) in STREAM_ERRORS:
            diagnostics['stream_error'] = str(error)
        raise
    finally:
        diagnostics['elapsed_seconds'] = time.monotonic()-started
        connection.close()


def main():
    diagnostics = {}
    try:
        frame = exchange(sys.stdin.buffer.read(MAX_BYTES + 1), diagnostics)
        sys.stdout.buffer.write(frame)
        return 0
    except Exception as error:
        # No traceback: stdin/native exception details might contain credentials.
        diagnostics.update(tls_diagnostics(error))
        sys.stderr.buffer.write((failure_code(error) + "\n").encode("ascii"))
        return 1
    finally:
        sys.stderr.buffer.write(b'DIAG ' + json.dumps(safe_diagnostics(diagnostics)).encode('ascii') + b'\n')


if __name__ == "__main__":
    raise SystemExit(main())
