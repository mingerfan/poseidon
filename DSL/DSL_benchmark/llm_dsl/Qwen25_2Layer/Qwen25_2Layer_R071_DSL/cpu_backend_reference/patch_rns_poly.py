"""Second isolated candidate: RNSPoly scalar-NTT paths bypass the generic wrappers."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);a=p.parse_args()
source=a.source.resolve();assert source.parent.parent==Path('/data/Workspace/gy/qwen-ckks-artifacts')
target=source/'src/poseidon/rns_poly.cpp';before=target.read_bytes();text=target.read_text()
assert 'QWEN_RNS_POLY_PARALLEL' not in text
text='#if defined(POSEIDON_CPU_RNS_PARALLEL) && defined(_OPENMP)\n#include <omp.h>\n#include <algorithm>\n#endif\n'+text
changes=[]
def span(signature):
    assert text.count(signature)==1,signature
    start=text.index('{',text.index(signature));depth=1;end=start+1
    while depth:
        if text[end]=='{':depth+=1
        elif text[end]=='}':depth-=1
        end+=1
    return start+1,end-1
for method,fn in (('coeff_to_dot','ntt_negacyclic_harvey'),('coeff_to_dot_lazy','ntt_negacyclic_harvey_lazy'),
                  ('dot_to_coeff','inverse_ntt_negacyclic_harvey'),('dot_to_coeff_lazy','inverse_ntt_negacyclic_harvey_lazy')):
    left,right=span('void RNSPoly::'+method+'()');old=text[left:right]
    assert old.count(fn+'(')==2 and old.count('for (')==2
    new='''
#if defined(POSEIDON_CPU_RNS_PARALLEL) && defined(_OPENMP)
    // QWEN_RNS_POLY_PARALLEL: same integer transform, disjoint Q/P arrays.
    const auto ntt_table = crt_context_->small_ntt_tables();
    const std::size_t count = rns_num_q_ + rns_num_p_;
    const int team = std::max(1, std::min<int>(omp_get_max_threads(), static_cast<int>(count)));
    #pragma omp parallel for schedule(static) num_threads(team) if(team > 1 && count >= 4 && poly_degree_ >= 4096 && !omp_in_parallel())
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(count); ++index)
    {
        const std::size_t table_index = index < rns_num_q_ ? index : index - rns_num_q_ + rns_p_offset_;
        FUNCTION(data_ + index * poly_degree_, ntt_table[table_index]);
    }
#else
ORIGINAL
#endif
'''.replace('FUNCTION',fn).replace('ORIGINAL',old)
    text=text[:left]+new+text[right:];changes.append(method)
for signature in ('void RNSPoly::operator*=(const RNSPoly &poly)','void RNSPoly::operator*=(uint64_t scalar)',
                  'void RNSPoly::operator+=(const RNSPoly &poly)','void RNSPoly::operator+=(uint64_t scalar)'):
    left,right=span(signature);body=text[left:right];assert body.count('for (')==2
    for loop,count in [('for (auto i = 0; i < rns_num_q_; ++i)','rns_num_q_'),
                       ('for (auto i = rns_num_q_; i < rns_num_q_ + rns_num_p_; ++i)','rns_num_p_')]:
        assert body.count(loop)==1
        pragma='''#if defined(POSEIDON_CPU_RNS_PARALLEL) && defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(std::max(1, std::min<int>(omp_get_max_threads(), static_cast<int>(COUNT)))) if(COUNT >= 4 && poly_degree_ >= 4096 && !omp_in_parallel())
#endif
    '''.replace('COUNT',count)
        body=body.replace(loop,pragma+loop)
    text=text[:left]+body+text[right:];changes.append(signature)
target.write_text(text)
(source.parent/'rns-poly-patch.json').write_text(json.dumps(dict(file=str(target),before_sha256=hashlib.sha256(before).hexdigest(),
    after_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),functions=changes,
    scope='Four Q/P NTT paths and four pointwise operators; disjoint prime slices, serial fallback retained'),indent=2))
print(json.dumps(dict(patched=str(target),functions=len(changes))))
