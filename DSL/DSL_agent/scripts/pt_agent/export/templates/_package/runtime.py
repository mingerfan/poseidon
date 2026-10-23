"""Read public configuration, pack tensors, and register a structured computation."""

import json
from pathlib import Path

import numpy as np

from . import model
from .ops import RawLayout, Tensor
from .core.layout import FlatLayout, MuxLayout, pack_flat, pack_mux, unpack_flat, unpack_mux
from .core.refresh import RefreshBackend


ROOT = Path(__file__).resolve().parent.parent


def read_config():
    return tuple(json.loads((ROOT/path).read_text(encoding="utf-8")) for path in
                 ("config/model.json", "config/approximations.json", "interfaces/default.json"))


def layout(record):
    fields = dict(record["layout"])
    kind = fields.pop("kind")
    return MuxLayout(**fields) if kind == "mux" else FlatLayout(**fields)


def pack_arrays(arrays):
    _, _, interface = read_config()
    if len(arrays) != len(interface["inputs"]):
        raise ValueError("input count differs from program parameters")
    packed = []
    for array, entry in zip(arrays, interface["inputs"]):
        array = np.asarray(array, dtype=np.float64)
        if array.ndim != len(entry["shape"]) or list(array.shape[1:]) != entry["shape"][1:]:
            raise ValueError("input array shape differs from parameter")
        lay = layout(entry)
        pack = pack_mux if isinstance(lay, MuxLayout) else pack_flat
        packed.append(np.stack([pack(sample, lay) for sample in array]))
    return packed


def unpack_arrays(arrays, interface=None):
    interface = read_config()[2] if interface is None else interface
    if len(arrays) != len(interface["outputs"]):
        raise ValueError("output and layout counts differ")
    outputs = []
    for array, entry in zip(arrays, interface["outputs"]):
        lay = layout(entry)
        array = np.asarray(array)
        if isinstance(lay, FlatLayout):
            value = unpack_flat(array, lay)
        else:
            value = (unpack_mux(array, lay) if array.ndim == 1 else
                     np.stack([unpack_mux(sample, lay) for sample in array]))
        outputs.append(value)
    return outputs


def invoke(backend, weights):
    config, approximations, interface = read_config()
    active = (RefreshBackend(backend, **{key: config[key] for key in
              ("input_level", "boot_in_level", "boot_out_level")}) if config["refresh"] == "agent" else backend)
    inputs = [Tensor(active.input(i), RawLayout(tuple(item["shape"]), config["slots"]))
              for i, item in enumerate(interface["inputs"])]
    outputs = model.forward(active, weights, approximations, *inputs)
    active.output([item.value for item in outputs])
    return outputs


def register(hc, weights):
    from .hecate_backend import HecateBackend

    config, _, interface = read_config()

    def computation(*inputs):
        backend = HecateBackend(config["slots"], hc, inputs)
        invoke(backend, weights)
        values = [value.value for value in backend._outputs]
        return values[0] if len(values) == 1 else values

    computation.__name__ = config["name"]
    return hc.func(",".join("c" for _ in interface["inputs"]))(computation)


def run_numpy(weights, arrays):
    from .numpy_backend import NumpyBackend

    config = read_config()[0]
    packed = pack_arrays(arrays)
    backend = NumpyBackend(config["slots"], batch=len(packed[0]))
    backend.bind(packed)
    invoke(backend, weights)
    return unpack_arrays([value.value for value in backend._outputs])


def load_arrays(path, entries):
    if str(path).endswith(".npz"):
        with np.load(path, allow_pickle=False) as archive:
            return [archive[item["name"]] for item in entries]
    if len(entries) != 1:
        raise ValueError("multiple tensors require an NPZ archive keyed by interface name")
    return [np.load(path, allow_pickle=False)]


def save_arrays(path, arrays, entries):
    if len(arrays) == 1 and not str(path).endswith(".npz"):
        np.save(path, arrays[0], allow_pickle=False)
    else:
        np.savez(path, **{item["name"]: value for item, value in zip(entries, arrays)})
