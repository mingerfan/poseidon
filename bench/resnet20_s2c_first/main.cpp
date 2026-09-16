// Dedicated ResNet20 application entry for the validated S2C-first chain.
//
// Deliberately do not link the legacy application's gpu_ckks_runtime.cpp,
// gpu_resnet20_inference.cpp, gpu_relu.cpp, or main.cpp.  Until the validated
// implementation is promoted out of Doc/analysis, compile it as a library-like
// translation unit so this executable has one unambiguous scheduler.
#define POSEIDON_S2C_FIRST_RESNET20_LIBRARY
#define POSEIDON_S2C_FIRST_ENABLE_TIMING
#include "../../Doc/analysis/resnet20_joint_chain_20260911/bootstrap_network_precision.cpp"

int main(int argc,char **argv) {
    return run_s2c_first_resnet20_application(argc,argv);
}
