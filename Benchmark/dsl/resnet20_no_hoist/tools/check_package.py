"""Verify compiler artifacts, preserved GPU evidence, and optional CPU C++ gates."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
from prepare_assets import PACKAGE,digest
from audit_strict_native_hevm import audit

def check(cpp=False):
    manifest=json.loads((PACKAGE/'manifest.sha256.json').read_text())
    for name,expected in manifest.items():
        assert digest(PACKAGE/name)==expected, name
    hevm=PACKAGE/'artifacts/strict-native-v1.strict_native_resnet20.hevm'
    compiler=audit(str(hevm),str(PACKAGE/'source/strict_native_resnet20_20261015_v1.lowering.json'))
    log=(PACKAGE/'evidence/run-strict-native-dsl-full-gpu2-v3.log').read_text()
    native=(PACKAGE/'evidence/native_full_20261014.log').read_text()
    def vector(text,prefix):
        return [float(v) for v in next(row for row in text.splitlines() if row.startswith(prefix+' ')).split()[1:]]
    dsl=vector(log,'STRICT_DSL_LOGITS');reference=vector(native,'NETWORK_REFERENCE_LOGITS');encrypted=vector(native,'NETWORK_LOGITS')
    result=json.loads((PACKAGE/'evidence/strict_native_full_comparison_20261015.json').read_text())
    argmax=lambda values:max(range(len(values)),key=values.__getitem__)
    assert argmax(dsl)==argmax(reference)==argmax(encrypted)==3
    assert max(abs(a-b) for a,b in zip(dsl,reference))==result['dsl_plain_max']
    assert max(abs(a-b) for a,b in zip(dsl,encrypted))==result['dsl_native_max']
    assert log.count('PHASE baseline_S2C_no_hoist')==18 and log.count('PHASE baseline_C2S_no_hoist')==18
    assert log.count('BASELINE_RESULT legacy_tolerance_0_002=PASS ')==18
    assert log.count('FOLD_RESULT integration=PASS ')==18
    assert not re.search(r'ERROR|terminate|out of memory|integration=FAIL|legacy_tolerance_0_002=FAIL',log)
    assert 'ops=13420 bootstrap_calls=18 exact_plain=2510 exact_mul=513 exact_accumulate=1762' in log
    if cpp:
        runtime=PACKAGE/'runtime'
        with tempfile.TemporaryDirectory(prefix='resnet20-dsl-tests-') as temporary:
            for name,arguments in (
                ('test_native_encode_rules',[str(hevm),str(hevm)+'.native_scales',str(hevm)+'.native_encode_rules']),
                ('test_native_rescale_groups',[str(hevm),str(hevm)+'.native_rescales']),
                ('test_native_schedule_metadata',[str(hevm),str(runtime/'native_snapshot/Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt')])):
                binary=str(Path(temporary)/name)
                subprocess.run(['c++','-std=c++20','-O2',str(runtime/(name+'.cpp')),'-o',binary],check=True)
                subprocess.run([binary]+arguments,check=True)
    return dict(status='PASS',tracked_files=len(manifest),compiler_raw_ops=compiler['raw_operations'],
                inputs=27,bootstraps=18,reference_prediction=3,actual_prediction=3,
                exact_checks=4785,dsl_plain_max=result['dsl_plain_max'],dsl_native_max=result['dsl_native_max'],
                scope='packaged historical GPU evidence and current CPU gates; not a new GPU run')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpp',action='store_true')
    args=parser.parse_args();print(json.dumps(check(args.cpp),indent=2))
