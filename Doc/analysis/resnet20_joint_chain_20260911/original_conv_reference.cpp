// Reuse the original CHW reference and weight loader; never create its runtime.
// Linker section GC discards application GPU inference/security entry points.
#include "original_conv_reference.h"
#include "/home/liufuyao/Work/poseidon_gpu_other/resnet20-9.3/benchmark/resnet20_gpu/gpu_resnet20_inference.cpp"

ApplicationNetworkFixture load_application_network_fixture(int image_id) {
    using namespace poseidon::benchmark::resnet20_gpu;
    const auto weights=load_resnet20_weights();
    const auto affine=[&](int i) {return folded_batch_norm(weights.bn_bias.at(i),
        weights.bn_running_mean.at(i),weights.bn_running_var.at(i),weights.bn_weight.at(i));};
    ApplicationNetworkFixture result;
    result.image_id=image_id;
    result.true_label=load_cifar10_label(image_id);
    result.image=load_cifar10_image_chw(image_id,kBoundary);
    const auto stem_bn=affine(0);
    result.stem={32,32,3,16,1,weights.conv_weight.at(0),stem_bn.scale,stem_bn.bias};
    PlainTensor tensor{32,32,3,result.image};
    tensor=plain_conv_bn(tensor,16,1,result.stem.weights,stem_bn);
    result.stem_output=tensor.values;
    plain_relu_inplace(tensor);
    result.stem_relu_output=tensor.values;
    int channels=16,conv=1,bn=1;
    for(int stage=1;stage<=3;++stage)for(int block=0;block<3;++block) {
        const int output_channels=16<<(stage-1);
        const bool transition=stage>1 && block==0;
        ApplicationBlockFixture out;
        out.stage=stage;out.block=block;out.option_a_shortcut=transition;out.input=tensor.values;
        const auto first_bn=affine(bn++);
        out.conv1={tensor.h,tensor.w,channels,output_channels,transition?2:1,
            weights.conv_weight.at(conv++),first_bn.scale,first_bn.bias};
        auto shortcut=tensor;
        auto branch=plain_conv_bn(tensor,output_channels,out.conv1.stride,
            out.conv1.weights,first_bn);
        out.conv1_output=branch.values;
        plain_relu_inplace(branch);out.act1_output=branch.values;
        const auto second_bn=affine(bn++);
        out.conv2={branch.h,branch.w,output_channels,output_channels,1,
            weights.conv_weight.at(conv++),second_bn.scale,second_bn.bias};
        branch=plain_conv_bn(branch,output_channels,1,out.conv2.weights,second_bn);
        out.conv2_output=branch.values;
        if(transition)shortcut=plain_downsample_shortcut(shortcut);
        out.shortcut_output=shortcut.values;
        tensor=plain_add(std::move(branch),shortcut);out.residual_output=tensor.values;
        plain_relu_inplace(tensor);out.output=tensor.values;
        channels=output_channels;
        result.blocks.push_back(std::move(out));
    }
    result.linear_weight=weights.linear_weight;
    result.linear_bias=weights.linear_bias;
    result.reference_logits=plain_head_logits(tensor,weights);
    if(conv!=19 || bn!=19 || result.blocks.size()!=9)
        throw std::runtime_error("independent ResNet20 fixture topology mismatch");
    return result;
}

ApplicationConvFixture load_application_conv_fixture(int image_id,int conv_index) {
    using namespace poseidon::benchmark::resnet20_gpu;
    if(conv_index!=1 && conv_index!=2)throw std::runtime_error("only first BasicBlock convolutions supported");
    const auto weights=load_resnet20_weights();
    const auto affine=[&](int i) {return folded_batch_norm(weights.bn_bias.at(i),
        weights.bn_running_mean.at(i),weights.bn_running_var.at(i),weights.bn_weight.at(i));};
    PlainTensor image{32,32,3,load_cifar10_image_chw(image_id,kBoundary)};
    auto stem=plain_conv_bn(image,16,1,weights.conv_weight.at(0),affine(0));
    const auto bn=affine(conv_index);
    return {image_id,std::move(stem.values),weights.conv_weight.at(conv_index),bn.scale,bn.bias,conv_index};
}

std::vector<double> original_application_first_block(
    const ApplicationConvFixture &conv1,const ApplicationConvFixture &conv2) {
    using namespace poseidon::benchmark::resnet20_gpu;
    if(conv1.conv_index!=1 || conv2.conv_index!=2 || conv1.image_id!=conv2.image_id || conv1.stem!=conv2.stem)
        throw std::runtime_error("invalid first-block fixtures");
    PlainTensor input{32,32,16,conv1.stem};plain_relu_inplace(input);
    const auto shortcut=input;
    auto branch=plain_conv_bn(input,16,1,conv1.weights,{conv1.bn_scale,conv1.bn_bias});
    plain_relu_inplace(branch);
    branch=plain_conv_bn(branch,16,1,conv2.weights,{conv2.bn_scale,conv2.bn_bias});
    auto result=plain_add(std::move(branch),shortcut);plain_relu_inplace(result);
    return result.values;
}

std::vector<double> original_application_conv(
    const std::vector<double> &input,const ApplicationConvFixture &fixture) {
    using namespace poseidon::benchmark::resnet20_gpu;
    if (input.size()!=16*32*32) throw std::runtime_error("CHW reference expects 16x32x32");
    return plain_conv_bn(PlainTensor{32,32,16,input},16,1,fixture.weights,
        BatchNormAffine{fixture.bn_scale,fixture.bn_bias}).values;
}
