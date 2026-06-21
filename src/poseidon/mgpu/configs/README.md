# Static Schedule Config Templates

These JSON files are CPU-side execution templates for the multi-GPU static
schedule tools. They declare placement, topology, preflight checks, and the
communication backends that are expected to be available.

- `single_gpu.json`: compatibility template for a one-device run.
- `single_node_8gpu.json`: first real target for an 8-GPU node using CUDA peer
  communication.
- `cluster_4x8_preview.json`: planning template for a 4-node, 8-GPU-per-node
  cluster. It keeps `inter_node` false and `require_ready` false until a real
  inter-node transport backend exists.

Use a template with:

```bash
poseidon_mgpu_dacapo_hevm_dump \
  --hevm /path/to/model.hevm \
  --constants /path/to/model.cst \
  --config src/poseidon/mgpu/configs/single_node_8gpu.json \
  --write-summary-json /tmp/model-mgpu-preflight.json \
  --write-schedule /tmp/model-mgpu-schedule.mlir \
  --no-schedule
```

Command-line flags passed after `--config` override the file.
