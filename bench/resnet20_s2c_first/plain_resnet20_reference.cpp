#include "original_conv_reference.h"
#include "resnet20_weights.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <utility>

// Plain CHW oracle local to the dedicated S2C-first application.  Keeping it
// here prevents the legacy GPU inference translation unit (and therefore its
// scheduler/runtime dependencies) from entering this executable.
namespace {
namespace model=poseidon::benchmark::resnet20_gpu;

constexpr double kBoundary=40.0;
constexpr double kBatchNormEpsilon=1.0e-5;

struct ReluTree {
    int depth=0;
    std::vector<int> nodes{-1,0};
    int m=0,l=0;

    void merge(const ReluTree &left,const ReluTree &right,int split) {
        depth=std::max(left.depth,right.depth)+1;
        nodes.assign(std::size_t{1}<<(depth+1),-1);nodes[0]=-1;nodes[1]=split;
        for(int index=1;index<=(1<<(left.depth+1))-1;++index) {
            const int offset=1<<static_cast<int>(std::log2(index));
            nodes[index+offset]=left.nodes[index];
        }
        for(int index=1;index<=(1<<(right.depth+1))-1;++index) {
            const int offset=1<<static_cast<int>(std::log2(index));
            nodes[index+2*offset]=right.nodes[index];
        }
    }
};

int power_of_two(int exponent) {
    if(exponent<0 || exponent>=30)throw std::invalid_argument("invalid ReLU exponent");
    return 1<<exponent;
}

void build_odd_baby_tree(int degree,ReluTree &tree) {
    const int depth=static_cast<int>(std::ceil(std::log2(static_cast<double>(degree))));
    int best_total=10000,best_m=0,best_l=0;ReluTree best_tree;
    for(int l=1;power_of_two(l)-1<=degree;++l)for(int m=1;power_of_two(m-1)<degree;++m) {
        std::vector<std::vector<int>> costs(degree+1,std::vector<int>(depth+1));
        std::vector<std::vector<ReluTree>> trees(degree+1,std::vector<ReluTree>(depth+1));
        costs[1][1]=0;for(int odd=3;odd<=degree;odd+=2)costs[odd][1]=10000;
        for(int level=2;level<=depth;++level)for(int odd=1;odd<=degree;odd+=2) {
            if(odd<=power_of_two(l)-1 && odd<=power_of_two(level-1)) {
                costs[odd][level]=0;continue;
            }
            int best=10000;ReluTree candidate;
            for(int split_log=1;split_log<=m-1 && power_of_two(split_log)<odd &&
                split_log<level;++split_log) {
                const int split=power_of_two(split_log);
                const int value=costs[odd-split][level-1]+costs[split-1][level]+1;
                if(value<best) {
                    best=value;candidate.merge(trees[split-1][level],
                        trees[odd-split][level-1],split);
                }
            }
            costs[odd][level]=best;trees[odd][level]=std::move(candidate);
        }
        const int total=costs[degree][depth]+power_of_two(l-1)+m-2;
        if(total<best_total) {
            best_total=total;best_tree=trees[degree][depth];best_m=m;best_l=l;
        }
    }
    tree=std::move(best_tree);tree.m=best_m;tree.l=best_l;
}

int coefficient_count(int degree,const ReluTree &tree) {
    const int tree_size=1<<(tree.depth+1);
    std::vector<int> decomposed(tree_size,-1);decomposed[1]=degree;
    for(int level=1;level<=tree.depth;++level)
    for(int node=1<<level;node<(1<<(level+1));++node)
        decomposed[node]=node%2==0?tree.nodes[node/2]-1:
            decomposed[node/2]-tree.nodes[node/2];
    int count=0;for(int node=1;node<tree_size;++node)
        if(tree.nodes[node]==0)count+=decomposed[node]+1;
    return count;
}

std::vector<ReluTree> make_relu_trees() {
    std::vector<ReluTree> result(3);
    build_odd_baby_tree(15,result[0]);build_odd_baby_tree(15,result[1]);
    build_odd_baby_tree(27,result[2]);return result;
}

std::vector<std::vector<double>> load_relu_coefficients(
    const std::vector<ReluTree> &trees) {
    const auto path=std::filesystem::path(POSEIDON_GPU_RESNET20_SOURCE_DIR)/
        "data/resnet20/relu_param/d13.txt";
    std::ifstream stream(path);
    if(!stream)throw std::runtime_error("failed to open ReLU coefficients: "+path.string());
    const std::vector<int> degrees{15,15,27};std::vector<std::vector<double>> result;
    for(std::size_t component=0;component<degrees.size();++component) {
        std::vector<double> coefficients(coefficient_count(degrees[component],trees[component]));
        for(double &coefficient:coefficients)if(!(stream>>coefficient))
            throw std::runtime_error("ReLU coefficient file is truncated");
        result.push_back(std::move(coefficients));
    }
    for(double &x:result[0])x/=2.0;
    for(double &x:result[1])x/=1.7;
    for(double &x:result[2])x*=0.5;
    return result;
}

double evaluate_relu_component(double input,int degree,
    const std::vector<double> &coefficients,const ReluTree &tree) {
    const int tree_size=1<<(tree.depth+1);
    std::vector<int> decomposed(tree_size,-1),coefficient_start(tree_size,-1);
    decomposed[1]=degree;int next=1;
    for(int level=1;level<=tree.depth;++level)
    for(int node=1<<level;node<(1<<(level+1));++node)
        decomposed[node]=node%2==0?tree.nodes[node/2]-1:
            decomposed[node/2]-tree.nodes[node/2];
    for(int node=1;node<tree_size;++node)if(tree.nodes[node]==0) {
        coefficient_start[node]=next;next+=decomposed[node]+1;
    }
    std::vector<double> chebyshev(degree+1);chebyshev[0]=1;chebyshev[1]=input;
    for(int index=2;index<=degree;++index)
        chebyshev[index]=2*input*chebyshev[index-1]-chebyshev[index-2];
    std::function<double(int)> evaluate=[&](int node) {
        if(tree.nodes[node]==0) {
            int coefficient=coefficient_start[node];
            double result=chebyshev[1]*coefficients.at(coefficient);coefficient+=2;
            for(int index=3;index<=decomposed[node];index+=2) {
                result+=chebyshev[index]*coefficients.at(coefficient);coefficient+=2;
            }
            return result;
        }
        return chebyshev[tree.nodes[node]]*evaluate(2*node+1)+evaluate(2*node);
    };
    return evaluate(1);
}

double relu_reference(double input) {
    static const auto trees=make_relu_trees();
    static const auto coefficients=load_relu_coefficients(trees);
    const std::vector<int> degrees{15,15,27};double step=input;
    for(std::size_t i=0;i<degrees.size();++i)
        step=evaluate_relu_component(step,degrees[i],coefficients[i],trees[i]);
    return input*(step+0.5);
}

struct BatchNormAffine {
    std::vector<double> scale,bias;
};

struct PlainTensor {
    int h=0,w=0,c=0;
    std::vector<double> values;
};

BatchNormAffine folded_batch_norm(const std::vector<double> &bias,
    const std::vector<double> &mean,const std::vector<double> &variance,
    const std::vector<double> &weight) {
    if(bias.size()!=mean.size() || bias.size()!=variance.size() ||
        bias.size()!=weight.size())
        throw std::invalid_argument("ResNet20 batch-normalization shape mismatch");
    BatchNormAffine result;
    result.scale.resize(bias.size());result.bias.resize(bias.size());
    for(std::size_t channel=0;channel<bias.size();++channel) {
        result.scale[channel]=weight[channel]/
            std::sqrt(variance[channel]+kBatchNormEpsilon);
        result.bias[channel]=(bias[channel]-mean[channel]*result.scale[channel])/
            kBoundary;
    }
    return result;
}

PlainTensor plain_conv_bn(const PlainTensor &input,int out_channels,int stride,
    const std::vector<double> &weights,const BatchNormAffine &affine) {
    const int output_h=input.h/stride,output_w=input.w/stride;
    if(affine.scale.size()!=static_cast<std::size_t>(out_channels) ||
        affine.bias.size()!=static_cast<std::size_t>(out_channels) ||
        weights.size()!=static_cast<std::size_t>(out_channels*input.c*3*3))
        throw std::invalid_argument("plain ResNet20 convolution shape mismatch");
    PlainTensor output{output_h,output_w,out_channels,
        std::vector<double>(static_cast<std::size_t>(output_h*output_w*out_channels))};
    for(int oc=0;oc<out_channels;++oc)for(int oh=0;oh<output_h;++oh)
    for(int ow=0;ow<output_w;++ow) {
        double sum=0;
        for(int ic=0;ic<input.c;++ic)for(int kh=0;kh<3;++kh)
        for(int kw=0;kw<3;++kw) {
            const int ih=oh*stride+kh-1,iw=ow*stride+kw-1;
            if(ih<0 || ih>=input.h || iw<0 || iw>=input.w)continue;
            const auto input_index=(static_cast<std::size_t>(ic)*input.h+ih)*input.w+iw;
            const auto weight_index=((static_cast<std::size_t>(oc)*input.c+ic)*3+kh)*3+kw;
            sum+=input.values[input_index]*weights[weight_index];
        }
        output.values[(static_cast<std::size_t>(oc)*output_h+oh)*output_w+ow]=
            sum*affine.scale[oc]+affine.bias[oc];
    }
    return output;
}

void plain_relu_inplace(PlainTensor &tensor) {
    for(double &value:tensor.values)value=relu_reference(value);
}

PlainTensor plain_add(PlainTensor left,const PlainTensor &right) {
    if(left.h!=right.h || left.w!=right.w || left.c!=right.c ||
        left.values.size()!=right.values.size())
        throw std::invalid_argument("plain ResNet20 residual shape mismatch");
    for(std::size_t index=0;index<left.values.size();++index)
        left.values[index]+=right.values[index];
    return left;
}

PlainTensor plain_downsample_shortcut(const PlainTensor &input) {
    PlainTensor output{input.h/2,input.w/2,input.c*2,
        std::vector<double>(static_cast<std::size_t>(input.h/2*input.w/2*input.c*2),0)};
    const int channel_offset=input.c/2;
    for(int channel=0;channel<input.c;++channel)for(int row=0;row<output.h;++row)
    for(int col=0;col<output.w;++col)
        output.values[(static_cast<std::size_t>(channel+channel_offset)*output.h+row)*output.w+col]=
            input.values[(static_cast<std::size_t>(channel)*input.h+row*2)*input.w+col*2];
    return output;
}

std::vector<double> plain_head_logits(const PlainTensor &features,
    const model::ResNet20Weights &weights) {
    if(features.h!=8 || features.w!=8 || features.c!=64)
        throw std::invalid_argument("ResNet20 head expects 8x8x64 features");
    std::vector<double> averages(64),logits(10);
    for(int channel=0;channel<64;++channel)for(int row=0;row<8;++row)
    for(int col=0;col<8;++col)
        averages[channel]+=features.values[(static_cast<std::size_t>(channel)*8+row)*8+col]
            *kBoundary/64.0;
    for(int output=0;output<10;++output) {
        logits[output]=weights.linear_bias[output];
        for(int feature=0;feature<64;++feature)
            logits[output]+=weights.linear_weight[static_cast<std::size_t>(output)*64+feature]
                *averages[feature];
    }
    return logits;
}
} // namespace

double original_application_relu(double input) {
    return relu_reference(input);
}

ApplicationNetworkFixture load_application_network_fixture(int image_id) {
    const auto weights=model::load_resnet20_weights();
    const auto affine=[&](int i) {return folded_batch_norm(weights.bn_bias.at(i),
        weights.bn_running_mean.at(i),weights.bn_running_var.at(i),weights.bn_weight.at(i));};
    ApplicationNetworkFixture result;
    result.image_id=image_id;result.true_label=model::load_cifar10_label(image_id);
    result.image=model::load_cifar10_image_chw(image_id,kBoundary);
    const auto stem_bn=affine(0);
    result.stem={32,32,3,16,1,weights.conv_weight.at(0),stem_bn.scale,stem_bn.bias};
    PlainTensor tensor{32,32,3,result.image};
    tensor=plain_conv_bn(tensor,16,1,result.stem.weights,stem_bn);
    result.stem_output=tensor.values;plain_relu_inplace(tensor);
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
        out.conv1_output=branch.values;plain_relu_inplace(branch);out.act1_output=branch.values;
        const auto second_bn=affine(bn++);
        out.conv2={branch.h,branch.w,output_channels,output_channels,1,
            weights.conv_weight.at(conv++),second_bn.scale,second_bn.bias};
        branch=plain_conv_bn(branch,output_channels,1,out.conv2.weights,second_bn);
        out.conv2_output=branch.values;
        if(transition)shortcut=plain_downsample_shortcut(shortcut);
        out.shortcut_output=shortcut.values;
        tensor=plain_add(std::move(branch),shortcut);out.residual_output=tensor.values;
        plain_relu_inplace(tensor);out.output=tensor.values;
        channels=output_channels;result.blocks.push_back(std::move(out));
    }
    result.linear_weight=weights.linear_weight;result.linear_bias=weights.linear_bias;
    result.reference_logits=plain_head_logits(tensor,weights);
    if(conv!=19 || bn!=19 || result.blocks.size()!=9)
        throw std::runtime_error("independent ResNet20 fixture topology mismatch");
    return result;
}

ApplicationConvFixture load_application_conv_fixture(int image_id,int conv_index) {
    if(conv_index!=1 && conv_index!=2)
        throw std::runtime_error("only first BasicBlock convolutions supported");
    const auto weights=model::load_resnet20_weights();
    const auto affine=[&](int i) {return folded_batch_norm(weights.bn_bias.at(i),
        weights.bn_running_mean.at(i),weights.bn_running_var.at(i),weights.bn_weight.at(i));};
    PlainTensor image{32,32,3,model::load_cifar10_image_chw(image_id,kBoundary)};
    auto stem=plain_conv_bn(image,16,1,weights.conv_weight.at(0),affine(0));
    const auto bn=affine(conv_index);
    return {image_id,std::move(stem.values),weights.conv_weight.at(conv_index),
        bn.scale,bn.bias,conv_index};
}

std::vector<double> original_application_conv(const std::vector<double> &input,
    const ApplicationConvFixture &fixture) {
    if(input.size()!=16*32*32)throw std::runtime_error("CHW reference expects 16x32x32");
    return plain_conv_bn(PlainTensor{32,32,16,input},16,1,fixture.weights,
        BatchNormAffine{fixture.bn_scale,fixture.bn_bias}).values;
}

std::vector<double> original_application_first_block(
    const ApplicationConvFixture &conv1,const ApplicationConvFixture &conv2) {
    if(conv1.conv_index!=1 || conv2.conv_index!=2 || conv1.image_id!=conv2.image_id ||
        conv1.stem!=conv2.stem)throw std::runtime_error("invalid first-block fixtures");
    PlainTensor input{32,32,16,conv1.stem};plain_relu_inplace(input);
    const auto shortcut=input;
    auto branch=plain_conv_bn(input,16,1,conv1.weights,
        BatchNormAffine{conv1.bn_scale,conv1.bn_bias});
    plain_relu_inplace(branch);
    branch=plain_conv_bn(branch,16,1,conv2.weights,
        BatchNormAffine{conv2.bn_scale,conv2.bn_bias});
    auto result=plain_add(std::move(branch),shortcut);plain_relu_inplace(result);
    return result.values;
}
