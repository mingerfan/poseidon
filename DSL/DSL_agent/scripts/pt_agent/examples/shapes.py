"""Shape-only metadata for the two R1 sample programs."""


def mlp_weight_shapes() -> dict:
    return {"fc1.weight": (32, 64), "fc1.bias": (32,), "fc2.weight": (10, 32), "fc2.bias": (10,)}


def resnet20_weight_shapes() -> dict:
    """Return 97 static shapes; this creates no model weights."""
    weights = {"conv1.weight": (16, 3, 3, 3)}

    def bn(prefix, channels):
        for name in ("weight", "bias", "running_mean", "running_var"):
            weights[f"{prefix}.{name}"] = (channels,)

    bn("bn1", 16)
    for stage, channels in ((1, 16), (2, 32), (3, 64)):
        for block in range(3):
            prefix = f"layer{stage}.{block}"
            previous = channels//2 if stage > 1 and block == 0 else channels
            weights[prefix + ".conv1.weight"] = (channels, previous, 3, 3)
            weights[prefix + ".conv2.weight"] = (channels, channels, 3, 3)
            bn(prefix + ".bn1", channels)
            bn(prefix + ".bn2", channels)
    weights["linear.weight"] = (10, 64)
    weights["linear.bias"] = (10,)
    return weights
