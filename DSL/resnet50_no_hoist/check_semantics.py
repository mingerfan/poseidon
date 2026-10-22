"""Compare packed DSL arithmetic with FP64 Torch and the native C++ CPU oracle."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ResNet50 import (BOUNDARY, SLOTS, dense, dense_pool, layout, load_image,
                      load_label, load_model, make_pool_relu, make_relu, network, pack_input)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('weights','relu-coefficients','values','labels','native-logits','report','runtime-fixture'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.report.exists() or args.runtime_fixture.exists():
        parser.error('use new output paths')
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    model,image = load_model(args.weights),load_image(args.values)
    label = load_label(args.labels)
    relu,pool_relu = make_relu(args.relu_coefficients,float),make_pool_relu(args.relu_coefficients,float)
    # Negative borders distinguish valid-candidate maxpool from zero padding.
    negative = -torch.arange(1,65,dtype=torch.float64).reshape(1,1,8,8)/256
    negative_error = float((dense_pool(negative,lambda x:torch.clamp_min(x,0))-
                            F.max_pool2d(negative,3,2,1)).abs().max())
    assert negative_error == 0
    errors = {}
    with torch.no_grad():
        reference,expected = dense(model,image,relu,pool_relu)
        pool_error = float((expected['stem.maxpool']-F.max_pool2d(expected['stem.relu'],3,2,1)).abs().max())
        assert pool_error <= 2e-4, pool_error
        native = json.loads(args.native_logits.read_text())['logits']
        assert len(native)==1000
        native_error = float((reference-torch.tensor(native)).abs().max())
        assert native_error <= 1e-8, native_error
        print('dense/native max_error',native_error,'pool approximation',pool_error,flush=True)
        def observe(name,values,shape):
            target = expected[name][0] if expected[name].ndim==4 else expected[name].reshape(-1,1,1)
            actual = layout.unpack(values,shape)[:,:target.shape[1],:target.shape[2]]
            assert torch.isfinite(actual).all(),name
            errors[name] = float((actual-target).abs().max())
            print(name,'max_error',errors[name],flush=True)
            assert errors[name] <= 1e-8,(name,errors[name])
        packed = pack_input(image)
        output = network(model,packed,relu,pool_relu,observe,clear=True)[0][:1000]
    dense_error = float((output-reference).abs().max())
    assert torch.isfinite(output).all() and dense_error<=1e-8,dense_error
    assert int(output.argmax())==int(reference.argmax())==label
    sources = {p.name:sha(p) for p in sorted(args.weights.glob('*.txt'))}
    sources.update(d13=sha(args.relu_coefficients),frontend=sha(Path(__file__).with_name('ResNet50.py')),
                   layout=sha(Path(__file__).with_name('imagenet_layout.py')))
    model_sha = hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    report = dict(status='PASS',scope='FP64 packed DSL/Torch/native C++ CPU; not CKKS execution',
                  model='resnet50',weights=267,boundary=BOUNDARY,slots=SLOTS,input_packs=len(packed),
                  model_sha256=model_sha,source_sha256=sources,image=0,label=label,prediction=int(output.argmax()),
                  stage_max_errors=errors,dense_max_error=dense_error,native_cpu_max_error=native_error,
                  maxpool_approximation_error=pool_error,negative_border_error=negative_error,logits=output.tolist(),
                  gpu_executed=False)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    fixture = dict(format_version=1,model='resnet50',seed=0,model_sha256=model_sha,
                   input=image.reshape(-1).tolist(),packed_inputs=[p.tolist() for p in packed],
                   input_encoding='frontend-packed-slots',python_output=output.tolist(),native_output=native,
                   label=label,output_encoding='actual-logits-first-1000-slots',bootstrap_requirement='native-no-hoist',
                   semantic_report_sha256=sha(args.report))
    args.runtime_fixture.write_text(json.dumps(fixture,allow_nan=False)+'\n')
    print('PASS',dense_error,'prediction',int(output.argmax()),flush=True)


if __name__=='__main__':
    main()
