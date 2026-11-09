"""Offline replay of saved selector replies after exact request comparison."""

import copy
import json
from pathlib import Path


class ReplayMismatch(RuntimeError):
    pass


def differences(expected, actual, path='$'):
    if isinstance(expected, dict) and isinstance(actual, dict):
        return [row for key in sorted(set(expected) | set(actual)) for row in (
            [dict(path=path+'.'+key, reason='missing', expected=key in expected, actual=key in actual)]
            if key not in expected or key not in actual else differences(expected[key], actual[key], path+'.'+key))]
    if isinstance(expected, list) and isinstance(actual, list):
        result = [] if len(expected) == len(actual) else [dict(path=path, reason='length', expected=len(expected), actual=len(actual))]
        return result+[row for i, (left, right) in enumerate(zip(expected, actual))
                       for row in differences(left, right, path+f'[{i}]')]
    return [] if expected == actual else [dict(path=path, reason='value', expected=expected, actual=actual)]


class SavedTuneClient:
    def __init__(self, directory, output, *, iteration, limit):
        self.directory, self.output = Path(directory), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.limit, self.used, self.index = limit, 0, 0
        all_requests = sorted(self.directory.glob('call-*.request.json'))
        self.requests = [p for p in all_requests if json.loads(p.read_text()).get('purpose', '').startswith(
            f'tune-select-iter{iteration}-round')]
        first = all_requests.index(self.requests[0]) if self.requests else 0
        self.used = sum(self._attempts(p) for p in all_requests[:first])

    def _attempts(self, request):
        stem = request.name.removesuffix('.request.json')
        count = len(list((self.directory/'attempts').glob(stem+'-*.record.json')))
        return max(1, count)

    @property
    def calls(self):
        return self.used

    def chat(self, messages, *, purpose):
        number = self.index+1
        actual = dict(messages=messages, purpose=purpose)
        request = self.requests[self.index] if self.index < len(self.requests) else None
        expected = json.loads(request.read_text()) if request is not None else None
        diff = differences(expected, actual)
        reply_path = request.with_name(request.name.replace('.request.json', '.response.json')) if request else None
        report = dict(request=str(request) if request else None, equal=not diff,
                      differences=diff, api_calls=0, response_exists=bool(reply_path and reply_path.exists()))
        (self.output/f'{number:03d}-request.actual.json').write_text(json.dumps(actual, ensure_ascii=False, indent=2)+'\n')
        (self.output/f'{number:03d}-comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        if diff or not report['response_exists']:
            raise ReplayMismatch('Saved tuning request differs or its reply is unavailable: '+str(request))
        self.index += 1
        self.used += self._attempts(request)
        return copy.deepcopy(json.loads(reply_path.read_text()))
