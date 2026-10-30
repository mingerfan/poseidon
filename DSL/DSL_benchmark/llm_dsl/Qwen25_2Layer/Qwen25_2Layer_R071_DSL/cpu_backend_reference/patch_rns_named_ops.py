"""Candidate: parallelize disjoint prime slices in named RNS operations."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
args = parser.parse_args()
source = args.source.resolve()
assert source.parent.parent == Path('/data/Workspace/gy/qwen-ckks-artifacts')
target = source / 'src/poseidon/rns_poly.cpp'
before = target.read_bytes()
text = target.read_text()
assert 'QWEN_RNS_POLY_PARALLEL' in text
assert 'QWEN_RNS_NAMED_PARALLEL' not in text
changes = []
for signature, expected in (
    ('void RNSPoly::multiply(const RNSPoly &operand, RNSPoly &result) const', 4),
    ('void RNSPoly::add(const RNSPoly &operand, RNSPoly &result) const', 4),
    ('void RNSPoly::multiply_scalar(uint64_t scalar, RNSPoly &result) const', 2),
    ('void RNSPoly::add_scalar(uint64_t scalar, RNSPoly &result) const', 2),
):
    assert text.count(signature) == 1
    left = text.index('{', text.index(signature)) + 1
    end = left
    depth = 1
    while depth:
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
        end += 1
    right = end - 1
    body = text[left:right]
    assert body.count('for (') == expected
    count = 0
    for loop, size in (
        ('for (int i = 0; i < rns_num_q_; ++i)', 'rns_num_q_'),
        ('for (int i = 0; i < rns_num_p_; ++i)', 'rns_num_p_'),
        ('for (auto i = 0; i < rns_num_q_; ++i)', 'rns_num_q_'),
        ('for (auto i = rns_num_q_; i < rns_num_q_ + rns_num_p_; ++i)', 'rns_num_p_'),
    ):
        occurrences = body.count(loop)
        assert occurrences <= 1
        if not occurrences:
            continue
        pragma = '''#if defined(POSEIDON_CPU_RNS_PARALLEL) && defined(_OPENMP)
    // QWEN_RNS_NAMED_PARALLEL: preserve arithmetic, offsets and per-prime order.
    #pragma omp parallel for schedule(static) num_threads(std::max(1, std::min<int>(omp_get_max_threads(), static_cast<int>(COUNT)))) if(COUNT >= 4 && poly_degree_ >= 4096 && !omp_in_parallel())
#endif
    '''.replace('COUNT', size)
        body = body.replace(loop, pragma + loop)
        count += 1
    assert count == expected
    text = text[:left] + body + text[right:]
    changes.append(dict(function=signature, loops=count))
target.write_text(text)
(source.parent / 'rns-named-patch.json').write_text(json.dumps(dict(
    before_sha256=hashlib.sha256(before).hexdigest(),
    after_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), functions=changes,
    scope='Named add/multiply and scalar variants; unchanged integer formulas and offsets'), indent=2))
print(json.dumps(dict(functions=len(changes), loops=sum(c['loops'] for c in changes))))
