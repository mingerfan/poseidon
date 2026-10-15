// Independent original CHW model; no ciphertext/runtime operations are invoked.
#include "resnet18/gpu_resnet18_inference.cpp"
#include "resnet50/gpu_resnet50_inference.cpp"
#include "native_plain_head50.h"
#include <fstream>
#include <iomanip>
int main(int argc,char **argv){
    using namespace poseidon::benchmark;
    if(argc!=4)throw std::runtime_error("usage: native_plain_oracle 18|50 image_id output.f64");
    const int model=std::stoi(argv[1]);const auto image_id=std::stoull(argv[2]);
    std::vector<double> logits;int label=-1;
    if(model==18){
        auto weights=resnet18_gpu::load_resnet18_weights();
        auto topology=resnet18_gpu::make_resnet18_topology();
        auto image=resnet18_gpu::load_imagenet_image_chw(image_id,resnet18_gpu::kBoundary);
        auto features=resnet18_gpu::plain_resnet18_features(image,topology,weights);
        logits=resnet18_gpu::plain_head_logits(features,weights);
        label=resnet18_gpu::load_imagenet_label(image_id);
    }else if(model==50){
        auto weights=resnet50_gpu::load_resnet50_weights();
        auto topology=resnet50_gpu::make_resnet50_topology();
        auto image=resnet50_gpu::load_imagenet_image_chw(image_id,resnet50_gpu::kBoundary);
        auto features=resnet50_gpu::plain_resnet50_features(image,topology,weights);
        logits=native_plain_head50(features,weights);
        label=resnet50_gpu::load_imagenet_label(image_id);
    }else throw std::runtime_error("model must be 18 or 50");
    if(logits.size()!=1000)throw std::runtime_error("expected 1000 logits");
    for(double x:logits)if(!std::isfinite(x))throw std::runtime_error("nonfinite oracle logits");
    const int prediction=std::max_element(logits.begin(),logits.end())-logits.begin();
    std::ofstream file(argv[3],std::ios::binary);
    file.write(reinterpret_cast<const char*>(logits.data()),logits.size()*8);file.close();
    if(!file)throw std::runtime_error("oracle output write failed");
    std::cout<<"{\"status\":\"PASS\",\"scope\":\"original native CHW plaintext polynomial oracle, not encrypted accuracy\","
        <<"\"model\":\"resnet"<<model<<"\",\"image_id\":"<<image_id
        <<",\"prediction\":"<<prediction<<",\"true_label\":"<<label
        <<",\"classification_correct\":"<<(prediction==label?"true":"false")
        <<",\"gpu_executed\":false}\n";
}
