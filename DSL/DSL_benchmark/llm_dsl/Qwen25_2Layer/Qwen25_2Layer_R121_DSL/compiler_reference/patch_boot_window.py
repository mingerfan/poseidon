"""Unvalidated isolated compiler candidate; never edits the pinned compiler.

Normalize consumes a logical level only when factor differs from waterline.
This patch changes the DP segment budget, not physical parameters or Boot API.
"""
import argparse
import hashlib
import json
from pathlib import Path

PINNED_SHA256 = '9b4072646fa17bf79ad9adbc5361c5e2607fff3f291339ac60142d27c8c5d933'


def transform(raw):
    assert hashlib.sha256(raw).hexdigest() == PINNED_SHA256
    text = raw.decode('utf-8')
    newline = '\r\n' if '\r\n' in text else '\n'
    def lines(value):
        return value.replace('\n', newline)
    before = lines('''      // A bootstrap normalized back to the waterline already consumes one
      // level. Fresh inputs have the full initial budget.
      int64_t initialBudget = lastBootLevel;''')
    after = lines('''      // Bootstrap defaults to logical level zero and preserves factor.
      // normalize() inserts Rescale only when factor differs from waterline.
      // The remaining scale/level verifier and native contract stay unchanged.
      const int64_t normalizationCost = factor == waterline ? 0 : 1;
      const int64_t segmentBudget = lastBootLevel - normalizationCost;
      int64_t initialBudget = lastBootLevel;''')
    bound = 'start >= std::max<int64_t>(0, end - (lastBootLevel - 1));'
    assert text.count(before) == text.count(bound) == 1
    updated = text.replace(before, after).replace(bound,
        'start >= std::max<int64_t>(0, end - segmentBudget);')
    assert updated.count('const int64_t normalizationCost') == 1
    assert 'if (scale == waterline)' in updated
    assert 'if (type.getLevel() > safeLevel)' in updated
    output = updated.encode('utf-8')
    return output, dict(before_sha256=PINNED_SHA256,
        after_sha256=hashlib.sha256(output).hexdigest(), changed_regions=2,
        scope='Source patch candidate only; no compiler build, plan or CKKS acceptance')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    assert a.input.resolve() != a.output.resolve()
    assert not a.output.exists() and not a.report.exists()
    changed, report = transform(a.input.read_bytes())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as f:
        f.write(changed)
    with a.report.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
