"""Candidate parallel rescale and coefficient-wise RNS base conversion."""
import argparse
import hashlib
import json
from pathlib import Path

def body_span(text, signature):
    assert text.count(signature) == 1, signature
    left = text.index('{', text.index(signature)) + 1
    end, depth = left, 1
    while depth:
        if text[end] == '{': depth += 1
        elif text[end] == '}': depth -= 1
        end += 1
    return left, end - 1

def transform_rescale(text):
    signature = 'void RNSTool::divide_and_round_q_last_ntt_inplace('
    left, right = body_span(text, signature)
    body = text[left:right]
    marker = '    POSEIDON_ALLOCATE_GET_COEFF_ITER(temp, coeff_count_, pool);'
    assert body.count(marker) == 1
    start, end = body_span(body, '[&](auto I)')
    loop = body[start:end]
    # Keep the exact arithmetic and lazy reduction branches; only replace the
    # outer iterator's entries. Inner iterators J remain untouched.
    for old, new in (
        ('get<0>(I)', 'input[index]'),
        ('get<1>(I)', 'inv_q_last_mod_q_[index]'),
        ('get<2>(I)', 'base_q_->base()[index]'),
        ('get<3>(I)', 'rns_ntt_tables[index]'),
    ):
        assert old in loop
        loop = loop.replace(old, new)
    assert 'get<' not in loop.replace('get<0>(J)', '').replace('get<1>(J)', '')
    assert body[end + 1:].strip() == ');'
    branch = '''
#if defined(POSEIDON_CPU_RNS_PARALLEL) && defined(_OPENMP)
    // QWEN_RESCALE_PARALLEL: one fixed-capacity scratch slice per worker.
    // Keep capacity independent of current level to limit memory-pool size bins.
    const int qwen_capacity = std::max(1, std::min(omp_get_max_threads(), 32));
    const int qwen_team = std::min<int>(qwen_capacity, static_cast<int>(base_q_size - 1));
    if (qwen_team > 1 && base_q_size >= 5 && coeff_count_ >= 4096 && !omp_in_parallel())
    {
        auto qwen_scratch = allocate_poly(coeff_count_, static_cast<size_t>(qwen_capacity), pool);
        #pragma omp parallel for schedule(static) num_threads(qwen_team)
        for (std::int64_t index = 0; index < static_cast<std::int64_t>(base_q_size - 1); ++index)
        {
            CoeffIter temp(qwen_scratch.get() + static_cast<size_t>(omp_get_thread_num()) * coeff_count_);
LOOP
        }
        return;
    }
#endif
'''.replace('LOOP', loop)
    body = body.replace(marker, branch + marker)
    return text[:left] + body + text[right:]

def transform_conversion(text):
    left, right = body_span(text, 'void BaseConverter::fast_convert_array(')
    body = text[left:right]
    marker = '    POSEIDON_ALLOCATE_GET_STRIDE_ITER(temp, uint64_t, count, ibase_size, pool);'
    assert body.count(marker) == 1
    assert body.count('dot_product_mod(') == 1
    assert body.count('multiply_uint_mod(') == 1
    branch = '''
#if defined(POSEIDON_CPU_RNS_PARALLEL) && defined(_OPENMP)
    // QWEN_CONVERSION_PARALLEL: original temporary matrix, disjoint coefficient rows.
    // Complete all input reads before any output writes, preserving alias safety.
    const int qwen_team = std::max(1, std::min(omp_get_max_threads(), 32));
    if (qwen_team > 1 && count >= 4096 && !omp_in_parallel())
    {
        #pragma omp parallel num_threads(qwen_team)
        {
            #pragma omp for schedule(static)
            for (std::int64_t coefficient = 0; coefficient < static_cast<std::int64_t>(count); ++coefficient)
            {
                for (size_t base = 0; base < ibase_size; ++base)
                {
                    const auto &factor = ibase_.inv_punctured_prod_mod_base_array()[base];
                    const auto &modulus = ibase_.base()[base];
                    temp[coefficient][base] = factor.operand == 1
                        ? barrett_reduce_64(in[base][coefficient], modulus)
                        : multiply_uint_mod(in[base][coefficient], factor, modulus);
                }
            }
            // omp for's implicit barrier is required here.
            #pragma omp for schedule(static)
            for (std::int64_t coefficient = 0; coefficient < static_cast<std::int64_t>(count); ++coefficient)
            {
                for (size_t base = 0; base < obase_size; ++base)
                {
                    out[base][coefficient] = dot_product_mod(temp[coefficient],
                        base_change_matrix_[base].get(), ibase_size, obase_.base()[base]);
                }
            }
        }
        return;
    }
#endif
'''
    body = body.replace(marker, marker + '\n' + branch)
    return text[:left] + body + text[right:]

def transform(text, mode='combined'):
    assert 'QWEN_RESCALE_PARALLEL' not in text and 'QWEN_CONVERSION_PARALLEL' not in text
    if mode in ('combined', 'rescale'): text = transform_rescale(text)
    if mode in ('combined', 'conversion'): text = transform_conversion(text)
    return '#if defined(POSEIDON_CPU_RNS_PARALLEL) && defined(_OPENMP)\n#include <omp.h>\n#include <algorithm>\n#endif\n' + text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--mode', choices=('combined', 'rescale', 'conversion'), default='combined')
    args = parser.parse_args()
    source = args.source.resolve()
    assert source.parent.parent == Path('/data/Workspace/gy/qwen-ckks-artifacts')
    target = source / 'src/poseidon/basics/util/rns.cpp'
    before = target.read_bytes()
    target.write_text(transform(target.read_text(), args.mode))
    (source.parent / 'rescale-conversion-patch.json').write_text(json.dumps(dict(
        mode=args.mode, before_sha256=hashlib.sha256(before).hexdigest(),
        after_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        max_threads=32, rescale_scratch_bytes_at_n65536={str(n):n*65536*8 for n in (8,16,32)},
        scope='Rescale scratch split by worker; base conversion split by coefficient; serial fallback preserved'), indent=2))
    print(json.dumps(dict(mode=args.mode, file=str(target))))

if __name__ == '__main__': main()
