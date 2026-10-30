"""Patch only a fresh, isolated source copy; integer transforms retain ordering per prime."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);a=p.parse_args()
source=a.source.resolve();assert source.parent.parent==Path('/data/Workspace/gy/qwen-ckks-artifacts')
target=source/'src/poseidon/basics/util/ntt.h';old=target.read_bytes();text=target.read_text()
assert 'POSEIDON_CPU_RNS_PARALLEL' not in text
text='#if defined(POSEIDON_CPU_RNS_PARALLEL) && defined(_OPENMP)\n#include <omp.h>\n#include <algorithm>\n#endif\n'+text
changes=[]
for name in ('ntt_negacyclic_harvey_lazy','ntt_negacyclic_harvey','inverse_ntt_negacyclic_harvey_lazy','inverse_ntt_negacyclic_harvey'):
    pattern=r'POSEIDON_ITERATE\(iter\(operand, tables\), coeff_modulus_size,\s*\[&\]\(auto I\) \{ '+name+r'\(get<0>\(I\), get<1>\(I\)\); \}\);'
    matches=list(re.finditer(pattern,text));assert len(matches)==1,(name,len(matches))
    original=matches[0].group()
    replacement='''#if defined(POSEIDON_CPU_RNS_PARALLEL) && defined(_OPENMP)
    // Distinct primes own disjoint coefficient arrays. Tables are immutable.
    // Do not nest teams; small RNS inputs retain the serial fallback.
    const int team = std::min<int>(omp_get_max_threads(), static_cast<int>(coeff_modulus_size));
    #pragma omp parallel for schedule(static) num_threads(team) if(team > 1 && coeff_modulus_size >= 4 && operand.poly_modulus_degree() >= 4096 && !omp_in_parallel())
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(coeff_modulus_size); ++index)
    {
        FUNCTION(operand[index], tables[index]);
    }
#else
    ORIGINAL
#endif'''.replace('FUNCTION',name).replace('ORIGINAL',original)
    text=text[:matches[0].start()]+replacement+text[matches[0].end():];changes.append(name)
target.write_text(text)
(source.parent/'parallel-patch.json').write_text(json.dumps(dict(file=str(target),before_sha256=hashlib.sha256(old).hexdigest(),
    after_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),functions=changes,scope='Only four RNS NTT wrappers; per-prime integer operations unchanged'),indent=2))
print(json.dumps(dict(patched=str(target),functions=changes)))
