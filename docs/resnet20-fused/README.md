# GPU ResNet20 functional bootstrapping experiment

This branch is based on `a0ca0a894746956550afce68609a1694fc41b236` and carries the independent A800 D31 experiment. D31 uses N=32768, Q20/P1 and 19 fused activations. Coefficients are under `trident-functional/resnet20/functional_params`; the application source is Trident `gs/feat-qwen/resnet20`.

On A800, encrypted predictions matched the frozen D31 plaintext oracle for 1000/1000 images; maximum logit error was 0.0002190412116. Accuracy against labels was 904/1000, versus 914/1000 for plaintext exact ReLU. Per-image results are in `d31-a800-1000-results.csv`. A800 image 0 D31 online wall median was 4.235 s (two runs); the frozen degree-22 GPU baseline was 2.954 s (three runs). Their algorithms and some CPU preparation differ. Historical V100 results are not measurements of this A800 version.

**SECURITY_NOT_VALIDATED:** an illustrative LWE model using actual QP, H=32, CBD(21), 16 positive and 16 negative secret entries and 32768 assumed samples produced an estimated primal-uSVP cost around 2^118.4 using the lattice-estimator default MATZOV model. Ring/sample and evaluation-key correlations are not fully modeled. This configuration must not be claimed to meet 128-bit security.

The H100 scripts in `scripts/resnet20-fused/` support cross-building and future real-device benchmarking. The sm_90 build was cross-compiled, but no H100 hardware was available and no H100 performance was measured. Configure source root and GMP paths as described in the scripts. The profiling and batch-test code is experimental; do not merge it into the stable branch without review.
