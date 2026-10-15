"""Expand the SHA-bound, losslessly compressed image0 benchmark fixtures."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

PACKAGE=Path(__file__).resolve().parents[1]
def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()

def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=json.loads((PACKAGE/'fixtures/strict-native-v1.data-report.json').read_text())
    for suffix,key in (('cst','cst'),('inputs.f64','input')):
        target=output/('strict-native-v1.'+suffix)
        if target.exists():
            if digest(target)!=report[key+'_sha256']:raise ValueError('preserve existing mismatched '+str(target))
            continue
        partial=target.with_suffix(target.suffix+'.partial')
        with gzip.open(PACKAGE/'fixtures'/('strict-native-v1.'+suffix+'.gz'),'rb') as source,partial.open('xb') as destination:
            shutil.copyfileobj(source,destination,1024*1024)
        if partial.stat().st_size!=report[key+'_bytes'] or digest(partial)!=report[key+'_sha256']:
            raise ValueError('expanded fixture SHA/size mismatch: '+str(partial))
        partial.rename(target)
    data_report=output/'strict-native-v1.data-report.json'
    contents=(PACKAGE/'fixtures/strict-native-v1.data-report.json').read_bytes()
    if data_report.exists() and data_report.read_bytes()!=contents:
        raise ValueError('preserve existing mismatched '+str(data_report))
    data_report.write_bytes(contents)
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(PACKAGE/'build/assets'))
    args=parser.parse_args();print(prepare(args.output))
