"""Create the user-requested no-hoist native baselines in isolated copies.

Only transform/key representations and the common 26GiB pool bound change.
The original repository and model arithmetic remain untouched.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

def prepare(root,output):
    hashes={}
    for model in (18,50):
        source=root/f'benchmark/resnet{model}_gpu';target=output/f'benchmark/resnet{model}_gpu'
        target.mkdir(parents=True,exist_ok=True)
        for file in source.iterdir():
            if file.suffix not in ('.h','.cpp'):continue
            shutil.copy2(file,target/file.name)
            hashes[str(file.relative_to(root))]=hashlib.sha256(file.read_bytes()).hexdigest()
    runtime=output/'benchmark/resnet50_gpu/gpu_ckks_runtime.cpp'
    original=runtime.read_text();text=original
    changes=[
        ('pool_(&upstream_, 1 << 20, std::nullopt)',
         'pool_(&upstream_, 1 << 20, std::size_t{26} << 30)'),
        ('std::unique_ptr<gpu::GpuLinearMatrixGroupQP> s2c_first_stc;',
         'std::unique_ptr<gpu::GpuLinearMatrixGroup> s2c_first_stc;'),
        ('std::unique_ptr<gpu::GpuLinearMatrixGroupQP> s2c_first_cts;',
         'std::unique_ptr<gpu::GpuLinearMatrixGroup> s2c_first_cts;'),
        ('auto uploaded_galois = gpu::GpuUploader::upload_double_hoist_galois_keys(',
         'auto uploaded_galois = gpu::GpuUploader::upload_galois_keys('),
        ('''    impl_->s2c_first_stc = std::make_unique<gpu::GpuLinearMatrixGroupQP>(
        gpu::GpuUploader::upload_linear_matrix_group_qp(
            slot_to_coeff, impl_->context, impl_->device_id, 1));''',
         '''    impl_->s2c_first_stc = std::make_unique<gpu::GpuLinearMatrixGroup>(
        gpu::GpuUploader::upload_linear_matrix_group(slot_to_coeff, impl_->device_id));'''),
        ('''    impl_->s2c_first_cts = std::make_unique<gpu::GpuLinearMatrixGroupQP>(
        gpu::GpuUploader::upload_linear_matrix_group_qp(
            coeff_to_slot, impl_->context, impl_->device_id, 1));''',
         '''    impl_->s2c_first_cts = std::make_unique<gpu::GpuLinearMatrixGroup>(
        gpu::GpuUploader::upload_linear_matrix_group(coeff_to_slot, impl_->device_id));'''),
        ('''    impl_->gpu_evaluator.dft_double_hoist(
        source,
        *impl_->s2c_first_stc,
        *impl_->gpu_bootstrap_galois_keys,
        *impl_->s2c_first_stc_workspace,
        s2c_output);''',
         '''    impl_->gpu_evaluator.dft_loose(
        source, *impl_->s2c_first_stc, *impl_->gpu_bootstrap_galois_keys, s2c_output);'''),
        ('''    impl_->gpu_evaluator.coeff_to_slot_double_hoist(
        raised,
        *impl_->s2c_first_cts,
        *impl_->s2c_first_minus_i,
        *impl_->gpu_bootstrap_galois_keys,
        *impl_->s2c_first_cts_workspace,
        real,
        imag);''',
         '''    impl_->gpu_evaluator.coeff_to_slot_loose(
        raised, *impl_->s2c_first_cts, *impl_->s2c_first_minus_i,
        *impl_->gpu_bootstrap_galois_keys, real, imag);'''),
        ('    std::cout << "S2C_FIRST_BOOTSTRAP_READY q=" << bootstrap_q_count',
         '    std::cout << "S2C_FIRST_BOOTSTRAP_READY linear_mode=no_hoist pool_GiB=26 q=" << bootstrap_q_count'),
        ('              << " output_q=" << impl_->config.application_q_count()',
         '              << " output_q=" << native_evalmod.output_q_count'),
    ]
    for old,new in changes:
        if text.count(old)!=1:raise ValueError(f'Native source layout changed: {old[:85]}')
        text=text.replace(old,new)
    assert 'upload_double_hoist_galois_keys(' not in text
    assert '.dft_double_hoist(' not in text and '.coeff_to_slot_double_hoist(' not in text
    runtime.write_text(text)
    (output/'native-nohoist-runtime.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True),text.splitlines(True),fromfile='original/gpu_ckks_runtime.cpp',tofile='nohoist/gpu_ckks_runtime.cpp')))
    (output/'native-nohoist-provenance.json').write_text(json.dumps({
        'scope':'isolated native no-hoist baseline, not DSL/GPU acceptance',
        'user_authorized':'both native and DSL must use no-hoist',
        'original_source_hashes':hashes,
        'nohoist_runtime_sha256':hashlib.sha256(runtime.read_bytes()).hexdigest(),
        'replacements':len(changes),'pool_gib':26,'headroom_gib':2,
        'coefficients_scales_rescale_schedule_topology':'unchanged'},indent=2)+'\n')
    print('Prepared isolated native no-hoist baselines; original native repository unchanged')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();prepare(args.root,args.output)
