"""Bind original rescale groups to verified real compiler instructions."""
import argparse
import json
from pathlib import Path
import struct
from verify_native_scale_sidecar import fnv
from audit_strict_native_hevm import audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hevm');parser.add_argument('manifest');parser.add_argument('output')
    args=parser.parse_args()
    audit(args.hevm,args.manifest)
    data=Path(args.hevm).read_bytes()
    _,header=struct.unpack_from('<II',data)
    body,nops=struct.unpack_from('<QQ',data,header)
    manifest=json.loads(Path(args.manifest).read_text())
    rows=[f'POSEIDON_HEVM_NATIVE_RESCALE_V1 {nops} {fnv(data):016x}']
    for group in manifest['rescale_groups']:
        count=group['count']
        if not count:continue
        begin,end=group['raw_begin'],group['raw_end']
        assert end-begin==2*count
        first=struct.unpack_from('<4H',data,header+body+8*(begin+1))
        last=struct.unpack_from('<4H',data,header+body+8*(end-1))
        mode=1 if group['native_rule'].startswith('ReLU') else 0
        rows.append(f'{begin} {end} {first[2]} {last[1]} {count} {mode} {group["origin"]}')
    Path(args.output).write_text('\n'.join(rows)+'\n')
    print(json.dumps(dict(status='PASS',rescale_groups=len(rows)-1,
                         scope='CPU bound schedule metadata; GPU dispatch pending')))
