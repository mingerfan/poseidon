"""Credential-free TLS route probe. GET /models only; never a generation call.

Explicit CONNECT tests the installed loopback proxy, not arbitrary destinations.
Short responses cannot certify long generation transfers or identify a bad-MAC
root cause. Each probe runs in a fresh process with a hard 20-second deadline.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import time

from deepseek_http_worker import tls_context, tls_diagnostics, failure_code
from workspace_paths import RESULTS


def probe(port):
    context = tls_context()
    connection = http.client.HTTPSConnection(
        '127.0.0.1' if port else 'api.deepseek.com',
        port=port or 443, timeout=12, context=context)
    if port:
        connection.set_tunnel('api.deepseek.com', 443)
    out = dict(route='loopback_connect' if port else 'transparent_default',
               proxy_port=port, python=sys.version.split()[0], openssl=ssl.OPENSSL_VERSION,
               certificate_required=context.verify_mode == ssl.CERT_REQUIRED,
               hostname_check=context.check_hostname, stage='connect')
    started = time.monotonic()
    try:
        connection.connect()
        out.update(tls_version=connection.sock.version(), cipher=connection.sock.cipher()[0],
                   certificate_sha256=hashlib.sha256(connection.sock.getpeercert(True)).hexdigest(),
                   stage='request')
        connection.request('GET', '/models', headers={'Accept': 'application/json'})
        response = connection.getresponse()
        out.update(status=response.status, stage='body')
        raw = response.read(8193)
        out.update(received_bytes=len(raw), stage='complete',
                   # An unauthorized GET is the expected credential-free result.
                   passed=response.status == 401 and len(raw) <= 8192)
    except Exception as error:
        out.update(passed=False, error=failure_code(error), **tls_diagnostics(error))
    finally:
        connection.close()
        out['elapsed_seconds'] = time.monotonic() - started
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proxy-port', type=int, default=7897)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.proxy_port <= 65535:
        parser.error('invalid proxy port')
    if args.child:
        print(json.dumps(probe(args.proxy_port)))
        return 0
    if not args.proxy_port:
        parser.error('parent requires a nonzero proxy port')
    def one(port):
        try:
            done = subprocess.run([sys.executable, '-B', __file__, '--child', '--proxy-port', str(port)],
                                  capture_output=True, timeout=20, check=True, text=True)
            return json.loads(done.stdout)
        except subprocess.TimeoutExpired:
            return dict(proxy_port=port, passed=False, error='hard_timeout')
        except (subprocess.SubprocessError, ValueError):
            return dict(proxy_port=port, passed=False, error='probe_process_failed')
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(one, [0, args.proxy_port] * 3))
    RESULTS.mkdir(parents=True, exist_ok=True)
    result = Path(tempfile.mkdtemp(prefix='deepseek-tls-probe-', dir=RESULTS))
    report = dict(paid_calls=0, credentials_loaded=False, probes=rows,
                  limitation='Short unauthenticated responses do not certify long paid responses.')
    (result / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('Evidence:', result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
