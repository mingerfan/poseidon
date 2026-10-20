"""Deterministic FP64 checks of multi-pack convolution, pooling and 1000-way FC."""
import numpy as np
import torch
import torch.nn.functional as F
import imagenet_layout as layout


def main():
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(123)
    for channels, output_channels, h, k, slots, stride, valid in (
        (3, 8, 8, 1, 128, 1, 7),
        (8, 16, 8, 1, 128, 2, 3),
        (16, 32, 4, 2, 128, 1, 3),
        (32, 64, 4, 2, 128, 2, 2),
    ):
        x = torch.from_numpy(rng.normal(size=(channels, h, h)))
        weight = rng.normal(size=(output_channels, channels, 3, 3))
        bias = rng.normal(size=output_channels)
        source = layout.shape(channels, h, k, slots)
        result, target = layout.conv(layout.pack(x, source), source, weight, bias,
                                      stride, valid, True)
        expected = F.conv2d(x[None], torch.from_numpy(weight), torch.from_numpy(bias),
                            stride=stride, padding=1)[0]
        expected[:, valid:, :] = 0
        expected[:, :, valid:] = 0
        error = float((layout.unpack(result, target)-expected).abs().max())
        print('conv', channels, output_channels, stride, 'error', error, flush=True)
        assert error < 1e-10
    x = torch.from_numpy(rng.normal(size=(16, 8, 8)))
    x[:, 6:, :] = 0
    x[:, :, 6:] = 0
    source = layout.shape(16, 8, 1, 128)
    result, target = layout.avg_pool(layout.pack(x, source), source, 3, True)
    expected = F.avg_pool2d(x[:, :6, :6][None], 3, 2, 1)[0]
    actual = layout.unpack(result, target)
    assert float((actual[:, :3, :3]-expected).abs().max()) < 1e-12
    print('avg_pool PASS', flush=True)
    source = layout.shape(512, 8, 16)
    x = torch.from_numpy(rng.normal(size=(512, 8, 8)))
    x[:, 7:, :] = 0
    x[:, :, 7:] = 0
    result = layout.global_pool(layout.pack(x, source), source, 7, True)
    expected = x[:, :7, :7].mean((1, 2))
    assert float((result[0][:512]-expected).abs().max()) < 1e-12
    weight, bias = rng.normal(size=(1000, 512)), rng.normal(size=1000)
    logits = layout.linear(result, weight, bias, 32768, True)[0][:1000]
    assert float((logits - F.linear(expected, torch.from_numpy(weight),
                                   torch.from_numpy(bias))).abs().max()) < 1e-10
    print('global_pool and 1000-way FC PASS', flush=True)


if __name__ == '__main__':
    main()
