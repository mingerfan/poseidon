"""Create an isolated metadata-only observer copy of native GpuUploader.

The native source is never edited. The observer only tags host metadata;
normal GPU upload/evaluation code is unchanged and is not executed here.
"""
import argparse
import hashlib
import json
from pathlib import Path

def prepare(source, output):
    original=source.read_bytes()
    text=original.decode().replace(chr(13)+chr(10), chr(10))
    needle='                result.meta.q_count = q_count;\n                return result;'
    if text.count(needle)!=1:
        raise ValueError('Native metadata-only encoder layout changed')
    text=text.replace(needle,'''                result.meta.q_count = q_count;
                result.poly_.poly_id = poseidon_bootstrap_export_scalar(value, q_count, scale);
                return result;''')
    declaration='''#include <complex>
#include <cstdint>
#include <cstddef>
std::uint64_t poseidon_bootstrap_export_scalar(std::complex<double>, std::size_t, double);
'''
    output.write_text(declaration+text)
    output.with_suffix('.provenance.json').write_text(json.dumps({
        'scope':'CPU metadata-only observer, not changed native arithmetic',
        'source_sha256':hashlib.sha256(original).hexdigest(),
        'observer_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'replacement_count':1},indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();prepare(args.source,args.output)
