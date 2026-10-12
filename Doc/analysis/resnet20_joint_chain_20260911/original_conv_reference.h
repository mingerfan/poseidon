#pragma once
#include <vector>

struct ApplicationConvFixture {
    int image_id;
    std::vector<double> stem, weights, bn_scale, bn_bias;
    int conv_index=1;
};

struct ApplicationLayerFixture {
    int input_h=0,input_w=0,input_channels=0,output_channels=0,stride=1;
    std::vector<double> weights,bn_scale,bn_bias;
};

struct ApplicationBlockFixture {
    int stage=0,block=0;
    bool option_a_shortcut=false;
    ApplicationLayerFixture conv1,conv2;
    // Independent CHW oracle values at every public block boundary.
    std::vector<double> input,conv1_output,act1_output,conv2_output;
    std::vector<double> shortcut_output,residual_output,output;
};

struct ApplicationNetworkFixture {
    int image_id=0,true_label=-1;
    std::vector<double> image;
    ApplicationLayerFixture stem;
    std::vector<double> stem_output,stem_relu_output;
    std::vector<ApplicationBlockFixture> blocks;
    std::vector<double> linear_weight,linear_bias,reference_logits;
};

ApplicationNetworkFixture load_application_network_fixture(int image_id);
ApplicationConvFixture load_application_conv_fixture(int image_id,int conv_index=1);
std::vector<double> original_application_conv(
    const std::vector<double> &input, const ApplicationConvFixture &fixture);
std::vector<double> original_application_first_block(
    const ApplicationConvFixture &conv1,const ApplicationConvFixture &conv2);
