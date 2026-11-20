"""Count the trusted selector layout without loading a model or fitting a sign.

The explicit coefficients are synthetic structure fixtures.  Counts and Python
resource observations describe this selector, while its polynomial quality,
model binding and CKKS memory remain unmeasured.
"""

import argparse
import json
from pathlib import Path
import time
import weakref


def count_selection(*, vocabulary=151936, slots=16384, degrees=(15, 15, 27)):
    import numpy as np
    from pt_agent.backends import CountBackend
    from pt_agent.kernels.encrypted_generation import select_maximum
    from pt_agent.kernels.layout import layout_blocks, multi_flat_layout

    class ObservedCount(CountBackend):
        def __init__(self):
            super().__init__(slots)
            self.live, self.peak_live = weakref.WeakSet(), 0

        def _track(self, value):
            self.live.add(value)
            self.peak_live = max(self.peak_live, len(self.live))
            return super()._track(value)

    coefficients = []
    for degree in degrees:
        if type(degree) is not int or degree < 1 or degree % 2 != 1:
            raise ValueError('Structure fixtures require positive odd degrees')
        array = np.zeros(degree+1, dtype=np.float64)
        array[1::2] = 1.0/(degree+1)
        coefficients.append(array)
    coefficients[-1] *= .5
    backend = ObservedCount()
    layout = multi_flat_layout(slots, vocabulary)
    inputs = [backend.input(index) for index, _ in enumerate(layout_blocks(layout))]
    started = time.perf_counter_ns()
    selected = select_maximum(backend, inputs, vocabulary, bound=1., normalized_gap=.01,
                             coefficients=coefficients)
    backend.output(selected['one_hot'])
    elapsed = time.perf_counter_ns()-started
    return dict(status='ok', vocabulary=vocabulary, slots=slots, capacity=slots//2,
        degrees=list(degrees), synthetic_coefficients=True,
        input_blocks=len(inputs), output_blocks=len(selected['one_hot']),
        selector_metadata=selected['metadata'], count_stats=backend.stats(),
        wall_seconds=elapsed/1e9, peak_live_cipher_handles=backend.peak_live,
        terminal_live_cipher_handles=len(backend.live),
        live_handle_slot_float64_payload_bytes=backend.peak_live*slots*8,
        live_handle_payload_scope='float64 slot payload for live arithmetic handles; CKKS ciphertext bytes unmeasured',
        mask_payload_scope='logical retained pattern inventory; CountBackend does not retain pattern arrays',
        original_model_loads=0, real_input_reads=0, embedding_weight_reads=0,
        polynomial_fits=0, api_calls=0, keygen_runs=0, ckks_executions=0,
        original_model_binding_verified=False, selector_quality_measured=False,
        compile_time_seconds=None, encrypted_latency_seconds=None,
        full_qwen_operation_counts=None, full_qwen_memory_bytes=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vocabulary', type=int, default=151936)
    parser.add_argument('--slots', type=int, default=16384)
    parser.add_argument('--degrees', type=int, nargs='+', default=(15, 15, 27))
    args = parser.parse_args(argv)
    result = count_selection(vocabulary=args.vocabulary, slots=args.slots, degrees=tuple(args.degrees))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('status','wall_seconds','peak_live_cipher_handles','input_blocks')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
