#!/usr/bin/env python3
"""Assert --compare actually injected and recovered from async-launch failure.

Run --compare with POSEIDON_GPU_KEYLOAD_TRACE=1 and pass its stderr path here.
Exit 77 instead of claiming coverage when the effective worker limit is one.
"""
import json
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text()
begin = 'QP_REGRESSION_ASYNC_FAILURE_BEGIN\n'
end = 'QP_REGRESSION_ASYNC_FAILURE_END\n'
assert text.count(begin) == text.count(end) == 1, 'missing/duplicate fault-test markers'
_, rest = text.split(begin, 1)
fault, after = rest.split(end, 1)

def profiles(s):
    prefix = 'QP_UPLOAD_PROFILE '
    return [json.loads(line[len(prefix):]) for line in s.splitlines() if line.startswith(prefix)]

injected = profiles(fault)
assert injected, 'no trace profiles in fault test'
if max(x['worker_limit'] for x in injected) <= 1:
    print('SKIP: effective worker limit is one; async launch was unavailable')
    raise SystemExit(77)
failures = sum(x['async_launch_failures'] for x in injected)
fallbacks = sum(x['serial_fallbacks'] for x in injected)
assert failures > 0 and fallbacks >= failures, 'fault was not observed and handled'
assert all(x['async_launched'] == 0 for x in injected), 'async work unexpectedly launched in fault test'
recovered = sum(x['async_launched'] for x in profiles(after))
assert recovered > 0, 'no later async work: worker slots may not have recovered'
print(f'PASS injected_async_failures={failures} serial_fallbacks={fallbacks} '
      f'later_async_launches={recovered}')
