"""Read public configuration, pack tensors, and register a structured computation."""

import json
from pathlib import Path

import numpy as np

from . import model
from .ops import RawLayout, Tensor
from .core.layout import FlatLayout, MuxLayout, pack_flat, pack_mux, unpack_flat, unpack_mux
from .core.refresh import RefreshBackend
from .core.layout import (MultiMuxLayout, MultiFlatLayout, layout_from_record,
                          layout_blocks, pack_multi, unpack_multi, TokenLayout, ScoreLayout,
                          pack_sequence,unpack_sequence)
from .core.multi import values_list


ROOT = Path(__file__).resolve().parent.parent


def read_config():
    return tuple(json.loads((ROOT/path).read_text(encoding="utf-8")) for path in
                 ("config/model.json", "config/approximations.json", "interfaces/default.json"))


def layout(record):
    return layout_from_record(record['layout'])


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
        if entry.get('kind')=='mask' and not np.isin(array,(0,1)).all():
            raise ValueError('Mask input values must be 0 or 1')
        if isinstance(lay,(TokenLayout,ScoreLayout)):
            blocks=[pack_sequence(sample,lay) for sample in array]
            packed.extend(np.stack([sample[i] for sample in blocks]) for i in range(len(lay.blocks)))
        elif isinstance(lay, (MultiMuxLayout, MultiFlatLayout)):
            blocks = [pack_multi(sample, lay) for sample in array]
            packed.extend(np.stack([sample[i] for sample in blocks]) for i in range(len(lay.blocks)))
        else:
            pack = pack_mux if isinstance(lay, MuxLayout) else pack_flat
            packed.append(np.stack([pack(sample, lay) for sample in array]))
    return packed


def unpack_arrays(arrays, interface=None):
    interface = read_config()[2] if interface is None else interface
    if len(arrays) != sum(len(entry.get('ciphertexts', [entry['ciphertext']])) for entry in interface['outputs']):
        raise ValueError("output and layout counts differ")
    outputs = []
    index = 0
    for entry in interface["outputs"]:
        lay = layout(entry)
        count = len(layout_blocks(lay))
        array = np.asarray(arrays[index])
        if isinstance(lay,(TokenLayout,ScoreLayout)):
            value=unpack_sequence(arrays[index:index+count],lay)
        elif isinstance(lay, (MultiMuxLayout, MultiFlatLayout)):
            value = unpack_multi(arrays[index:index+count], lay)
        elif isinstance(lay, FlatLayout):
            value = unpack_flat(array, lay)
        else:
            value = (unpack_mux(array, lay) if array.ndim == 1 else
                     np.stack([unpack_mux(sample, lay) for sample in array]))
        outputs.append(value)
        index += count
    return outputs


def invoke(backend, weights):
    config, approximations, interface = read_config()
    active = (RefreshBackend(backend, **{key: config[key] for key in
              ("input_level", "boot_in_level", "boot_out_level")}) if config["refresh"] == "agent" else backend)
    inputs, index = [], 0
    for item in interface['inputs']:
        count = len(layout_blocks(layout(item)))
        values = [active.input(index+i) for i in range(count)]
        index += count
        inputs.append(Tensor(values if count > 1 else values[0], RawLayout(tuple(item['shape']), config['slots'])))
    outputs = model.forward(active, weights, approximations, *inputs)
    active.output([value for item in outputs for value in
                   (item.value if isinstance(item.layout,(TokenLayout,ScoreLayout)) else values_list(item.value,item.layout))])
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
    return hc.func(",".join("c" for item in interface['inputs'] for _ in layout_blocks(layout(item))))(computation)


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


def ciphertext_entries(entries):
    """Distinct NPZ keys preserve every block when a tensor spans ciphertexts."""
    return [dict(name=(entry['name'] if len(entry.get('ciphertexts', [entry['ciphertext']])) == 1
                       else entry['name']+'__ct'+str(index)))
            for entry in entries for index in entry.get('ciphertexts', [entry['ciphertext']])]
