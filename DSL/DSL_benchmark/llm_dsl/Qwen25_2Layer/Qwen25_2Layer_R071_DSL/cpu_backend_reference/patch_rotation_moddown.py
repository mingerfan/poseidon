"""Candidate coefficient tiles for NTT permutation and final QP mod-down."""
import argparse
import hashlib
import json
from pathlib import Path

def span(text, signature):
    assert text.count(signature)==1,signature
    left=text.index('{',text.index(signature))+1;end=left;depth=1
    while depth:
        if text[end]=='{':depth+=1
        elif text[end]=='}':depth-=1
        end+=1
    return left,end-1

INCLUDES='''#if defined(POSEIDON_CPU_RNS_PARALLEL) && defined(_OPENMP)
#include <omp.h>
#include <algorithm>
#endif
'''

def rotation(text):
    assert 'QWEN_ROTATION_TILES' not in text
    left,right=span(text,'void apply_galois_ntt(ConstRNSIter operand,')
    body=text[left:right]
    anchor='        POSEIDON_ITERATE(iter(operand, result), coeff_modulus_size,'
    assert body.count(anchor)==1
    branch='''
#if defined(POSEIDON_CPU_RNS_PARALLEL) && defined(_OPENMP)
        // QWEN_ROTATION_TILES: initialize the shared immutable table before workers.
        const size_t qwen_tile_size = 4096;
        const size_t qwen_tiles = (coeff_count_ + qwen_tile_size - 1) / qwen_tile_size;
        const int qwen_team = std::min<int>(std::max(1, omp_get_max_threads()),
            static_cast<int>(std::min<size_t>(32, coeff_modulus_size * qwen_tiles)));
        if (qwen_team > 1 && coeff_count_ >= 4096 && !omp_in_parallel())
        {
#ifdef POSEIDON_DEBUG
            if (!(galois_elt & 1) || (galois_elt >= 2 * (uint64_t(1) << coeff_count_power_)))
                throw std::invalid_argument("Galois element is not valid");
            for (size_t i = 0; i < coeff_modulus_size; ++i)
                if (operand[i] == result[i])
                    throw std::invalid_argument("result cannot point to the same value as operand");
#endif
            generate_table_ntt(galois_elt, permutation_tables_[GetIndexFromElt(galois_elt)]);
            const auto qwen_table = permutation_tables_[GetIndexFromElt(galois_elt)].get();
            #pragma omp parallel for collapse(2) schedule(static) num_threads(qwen_team)
            for (std::int64_t modulus_index = 0; modulus_index < static_cast<std::int64_t>(coeff_modulus_size); ++modulus_index)
                for (std::int64_t tile = 0; tile < static_cast<std::int64_t>(qwen_tiles); ++tile)
                {
                    const size_t begin = static_cast<size_t>(tile) * qwen_tile_size;
                    const size_t end = std::min(coeff_count_, begin + qwen_tile_size);
                    for (size_t coefficient = begin; coefficient < end; ++coefficient)
                        result[modulus_index][coefficient] = operand[modulus_index][qwen_table[coefficient]];
                }
            return;
        }
#endif
'''
    body=body.replace(anchor,branch+anchor)
    result=text[:left]+body+text[right:]
    assert result.count('#pragma once')==1
    return result.replace('#pragma once','#pragma once\n'+INCLUDES,1)

def moddown(text):
    assert 'QWEN_MODDOWN_TILES' not in text
    left,right=span(text,'void RNSToolQP::mod_down(')
    body=text[left:right]
    anchor='    base_p_to_q_conv_->fast_convert_array(input_p, tmp, pool);'
    assert body.count(anchor)==1
    assert body.count('sub_poly_coeffmod(')==1 and body.count('multiply_poly_scalar_coeffmod(')==1
    branch='''
#if defined(POSEIDON_CPU_RNS_PARALLEL) && defined(_OPENMP)
    // QWEN_MODDOWN_TILES: original subtraction then scalar product on each tile.
    const size_t qwen_tile_size = 4096;
    const size_t qwen_tiles = (coeff_count_ + qwen_tile_size - 1) / qwen_tile_size;
    const int qwen_team = std::min<int>(std::max(1, omp_get_max_threads()),
        static_cast<int>(std::min<size_t>(32, base_q_size * qwen_tiles)));
    if (qwen_team > 1 && coeff_count_ >= 4096 && !omp_in_parallel())
    {
        #pragma omp parallel for collapse(2) schedule(static) num_threads(qwen_team)
        for (std::int64_t modulus_index = 0; modulus_index < static_cast<std::int64_t>(base_q_size); ++modulus_index)
            for (std::int64_t tile = 0; tile < static_cast<std::int64_t>(qwen_tiles); ++tile)
            {
                const size_t begin = static_cast<size_t>(tile) * qwen_tile_size;
                const size_t count = std::min(qwen_tile_size, coeff_count_ - begin);
                auto output = destination[modulus_index] + begin;
                sub_poly_coeffmod(input_q[modulus_index] + begin, tmp[modulus_index] + begin,
                    count, base_q_->base()[modulus_index], output);
                multiply_poly_scalar_coeffmod(output, count, prod_p_inv_mod_qi_[modulus_index].operand,
                    base_q_->base()[modulus_index], output);
            }
        return;
    }
#endif
'''
    body=body.replace(anchor,anchor+'\n'+branch)
    return INCLUDES+text[:left]+body+text[right:]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path)
    parser.add_argument('--mode',choices=('combined','rotation','moddown'),default='combined')
    args=parser.parse_args();source=args.source.resolve()
    assert source.parent.parent==Path('/data/Workspace/gy/qwen-ckks-artifacts')
    changes=[]
    for name,mode,fn in (('basics/util/galois.h','rotation',rotation),('util/rns_tool_qp.cpp','moddown',moddown)):
        if args.mode not in ('combined',mode):continue
        p=source/'src/poseidon'/name;before=p.read_bytes();p.write_text(fn(p.read_text()))
        changes.append(dict(file=name,before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (source.parent/'rotation-moddown-patch.json').write_text(json.dumps(dict(mode=args.mode,files=changes,tile_coefficients=4096,max_threads=32,
        scope='Disjoint coefficient tiles; table initialization before OpenMP; no added full-size scratch'),indent=2))
    print(json.dumps(dict(mode=args.mode,files=len(changes))))

if __name__=='__main__':main()
