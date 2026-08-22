# Local ResNet20 runtime data

This directory contains the runtime data needed by `poseidon_gpu_resnet20`:

- `resnet20/pretrained_parameters/resnet20_new`: trained convolution, batch
  normalization, and linear-layer parameters;
- `resnet20/testFile`: CIFAR-10 test values and labels used by the benchmark;
- `resnet20/relu_param/d13.txt`: polynomial ReLU coefficients.

The files were copied from this workspace's `Trident/resnet20` implementation
so that `benchmark/resnet20_gpu` does not depend on that external directory.
