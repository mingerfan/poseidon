"""Admit combined only after a component precision improvement, using stock profile generation."""
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent/'resnet18-model-v2'
FRAME = ROOT.parent/'resnet20-framework-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    paths = [ROOT.parent/'resnet18-precision-v1/component/ratio32.json',ROOT/'component/combined.json']
    results = []
    for path in paths:
        data = json.loads(path.read_text())
        errors = [a-b for a,b in zip(data['actual'],data['expected'])]
        assert len(errors)==32768 and all(math.isfinite(e) for e in errors)
        results.append(dict(max_abs=max(map(abs,errors)),rms=(sum(e*e for e in errors)/len(errors))**.5))
    assert results[1]['max_abs']<results[0]['max_abs']/4,results
    assert results[1]['rms']<results[0]['rms']/4,results
    def fields(path,prefix):
        row, = [line for line in path.read_text().splitlines() if line.startswith(prefix+' ')]
        return dict(item.split('=',1) for item in row.split()[1:])
    oldlog = ROOT.parent/'resnet18-precision-v1/component/ratio32.log'
    newlog = ROOT/'component/combined.log'
    assert fields(oldlog,'NATIVE_BOOT_CONTEXT')==fields(newlog,'NATIVE_BOOT_CONTEXT')
    meta = fields(newlog,'NATIVE_BOOT_PROFILE')
    assert meta['linear_mode']=='4' and meta['evalmod_degree']=='58' and meta['evalmod_log_scale']=='60'
    subprocess.run(['python3',str(ROOT/'generate_profiles.py'),'--runtime-root',
        str(FRAME/'poseidon/third_party/ckks-runtime'),'--native-probe-log',str(newlog),
        '--compiler-profile-template',str(FRAME/'profiles-native-v4/compiler-profile.json'),
        '--output',str(ROOT/'profiles-combined')],check=True)
    config_path = ROOT/'profiles-combined/native-config.json'
    config = json.loads(config_path.read_text())
    config['bootstrap_ratio'] = 128
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    shutil.copyfile(BASE/'compile_model.py',ROOT/'compile_model.py')
    report = dict(status='PASS',baseline_scale52=results[0],corrected_combined=results[1],
        max_error_improvement=results[0]['max_abs']/results[1]['max_abs'],
        rms_improvement=results[0]['rms']/results[1]['rms'],metadata=meta,
        bootstrap_ratio=128,degree=58,model_unchanged=True,original_accuracy_gates_unchanged=True)
    (ROOT/'component-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

    source_dir = ROOT/'runtime-support'
    source_dir.mkdir()
    source = source_dir/'poseidon_gpu_mlp_e2e_test.cpp'
    text = (BASE/'runtime-support/poseidon_gpu_mlp_e2e_test.cpp').read_text()
    marker = '                config.eval_mod_sine_degree = 58;'
    assert text.count(marker)==1
    text = text.replace(marker,marker+'\n                config.bootstrap_ratio = native_config.value("bootstrap_ratio",std::uint32_t{32});')
    source.write_text(text)
    old = next(p for p in json.loads((BASE/'runtime-build-final/provenance.json').read_text())
               if p['target']=='poseidon_runtime_gpu_mlp_e2e')
    out = ROOT/'runtime-build'
    out.mkdir()
    command,link = old['compile'][:],old['link'][:]
    original_source = command[command.index('-c')+1]
    original_object = command[command.index('-o')+1]
    command[command.index('-c')+1] = str(source)
    command[command.index('-o')+1] = str(out/'gpu_test.o')
    link[link.index(original_object)] = str(out/'gpu_test.o')
    binary = out/'poseidon_runtime_gpu_mlp_e2e'
    link[link.index('-o')+1] = str(binary)
    with (out/'build.log').open('x') as log:
        for args in (command,link):
            subprocess.run(args,cwd=old['cwd'],stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'provenance.json').write_text(json.dumps(dict(compile=command,link=link,cwd=old['cwd'],
        original_source=original_source,original_source_sha256=sha(Path(original_source)),
        source_sha256=sha(source),binary_sha256=sha(binary),
        change='Only read validated bootstrap_ratio from isolated Native configuration; no numeric gate changes'),indent=2)+'\n')
    shutil.copyfile(BASE/'compile_model.py',ROOT/'compile_model.py')
