"""Explicit stage-plan reuse for symbolic two-token generation.

This helper has no runner, model loader or planner implementation. Callers supply
the same frozen prefill callback/plan twice and two independently planned kernel
callbacks. Every stage aliases existing physical ciphertexts. Level5 admission
comes from an original input or an actual recorded bootstrap.
"""

from contextlib import contextmanager
from dataclasses import dataclass


VARIANT = 'full24-two-token-explicit-stage-plan-reuse-v1'
LEVELS = dict(input_level=5, boot_in_level=1, boot_out_level=5)
SEQUENCE = 32
INPUTS = 34


@dataclass(frozen=True, eq=False)
class Handle:
    value: object
    level: int
    serial: int


@contextmanager
def compact_scope():
    """Change only private dispatch guards; retain outer Refresh arithmetic."""
    from .qwen_prefill_compile import _projection_backend_scope, _projection_delegate, _projection_backend_allowed
    from ..kernels import compact_linear
    original = compact_linear.embedding_from_selection
    delegated = _projection_delegate(original, _projection_backend_allowed)
    try:
        compact_linear.embedding_from_selection = delegated
        with _projection_backend_scope(), compact_linear.compact_transformer_projections():
            yield
    finally:
        compact_linear.embedding_from_selection = original


class Segments:
    """A small boundary SSA ledger, separate from original backend arithmetic."""
    def __init__(self, backend):
        if backend.slots < SEQUENCE or backend._inputs or backend._outputs is not None:
            raise ValueError('Require a fresh physical backend with at least32slots')
        if hasattr(backend, 'input_level') and backend.input_level != 5:
            raise ValueError('Level-aware physical backend inputs must actually start at5')
        if hasattr(backend, 'boot_out_level') and backend.boot_out_level != 5:
            raise ValueError('Physical bootstrap must actually return level5')
        self.backend, self._known, self._busy, self.closed = backend, {}, False, False
        self.metadata = dict(variant=VARIANT, levels=dict(LEVELS), boundary_bootstraps=0,
                             physical_input_calls=0, segments=[], ssa=[], final_output_calls=0)
        self.inputs = []
        for index in range(INPUTS):
            self.inputs.append(self._issue(backend.input(index), 5, 'original_input', [], index=index))
            self.metadata['physical_input_calls'] += 1
        self._physical_inputs = backend._inputs
        self._original_values = tuple(backend._inputs)

    def _issue(self, value, level, kind, sources, **details):
        if (value.backend is not self.backend or type(level) is not int or not 1 <= level <= 5):
            raise ValueError('Require a physical ciphertext and its actual admissible level')
        actual = getattr(value, 'level', None)
        if actual is not None and actual != level:
            raise ValueError('Observed physical level differs from the boundary proof')
        serial = len(self._known)
        handle = Handle(value, level, serial)
        self._known[serial] = handle
        self.metadata['ssa'].append(dict(id=serial, kind=kind, level=level,
                                         sources=[item.serial for item in sources], **details))
        return handle

    def _require(self, handle, *, level=None):
        if (not isinstance(handle, Handle) or self._known.get(handle.serial) is not handle
                or handle.value.backend is not self.backend):
            raise ValueError('Require the original handle issued by this physical boundary ledger')
        if level is not None and handle.level != level:
            raise ValueError('Stage inputs require proved level5; emit a real boundary bootstrap')
        actual = getattr(handle.value, 'level', None)
        if actual is not None and actual != handle.level:
            raise ValueError('Physical level and recorded provenance differ')

    def _physical_unchanged(self):
        if (self.backend._inputs is not self._physical_inputs
                or len(self.backend._inputs) != INPUTS
                or any(a is not b for a, b in zip(self.backend._inputs, self._original_values))):
            raise ValueError('A segment changed the original34physical input handles')

    def normalize(self, handles, label):
        if self.closed or self._busy:
            raise ValueError('Boundary normalization occurs between active segments')
        result = []
        for handle in handles:
            self._require(handle)
            value = handle.value.bootstrap()
            result.append(self._issue(value, 5, 'boundary_bootstrap', [handle], label=label))
            self.metadata['boundary_bootstraps'] += 1
        return result

    def masks(self, prefix_length):
        """Construct actual ciphertext expressions for the public second mask."""
        import numpy as np
        if self.closed or self._busy or type(prefix_length) is not int or not 1 <= prefix_length <= 31:
            raise ValueError('Use a finite public prefix with one further padding position')
        source = self.inputs[0]
        self._require(source, level=5)
        zero = self._issue(source.value * self.backend.const([0.]), 4, 'public_zero_mul', [source])
        outputs = []
        for name in ('attention_mask', 'sel'):
            vector = np.zeros(self.backend.slots, dtype=np.float64)
            if name == 'attention_mask':
                vector[:prefix_length] = 1.
            else:
                vector[prefix_length-1] = 1.
            value = zero.value + self.backend.const(vector)
            outputs.append(self._issue(value, 4, 'public_mask_add', [zero], name=name,
                                       prefix_length=prefix_length))
        return outputs

    def run(self, name, inputs, plan, callback):
        """Use original Refresh.input/output, with physical I/O aliases only."""
        from ..backends import RefreshBackend
        if self.closed or self._busy:
            raise ValueError('Segments run serially before the single final output')
        if (plan.get('format') != 'pt-agent-refresh-plan-v1' or plan.get('levels') != LEVELS
                or type(plan.get('boots')) is not int or plan['boots'] < 0):
            raise ValueError('Require the unchanged explicit qwen48segment plan')
        inputs = list(inputs)
        for handle in inputs:
            self._require(handle, level=5)
        self._physical_unchanged()
        active = RefreshBackend(self.backend, policy='plan', plan=plan, **LEVELS)
        row = dict(name=name, status='running', inputs=[value.serial for value in inputs],
                   planned_calls=plan['calls'], planned_boots=plan['boots'], logical_input_calls=0)
        self.metadata['segments'].append(row)
        captured, declared = [], []
        output_called = False
        sentinel = object()
        previous = {key: self.backend.__dict__.get(key, sentinel) for key in ('input', 'output')}
        original_output = active.output

        def alias_input(index):
            if type(index) is not int or index != row['logical_input_calls'] or index >= len(inputs):
                raise ValueError('Segment logical inputs must retain their consecutive plan numbering')
            row['logical_input_calls'] += 1
            return inputs[index].value

        def alias_output(values):
            if captured:
                raise ValueError('Each segment must declare its output once')
            captured.extend(values)

        def observe_output(values):
            nonlocal output_called
            if output_called:
                raise ValueError('Each segment must call original Refresh.output exactly once')
            output_called = True
            declared.extend(values)
            original_output(declared)

        self._busy = True
        self.backend.input, self.backend.output, active.output = alias_input, alias_output, observe_output
        try:
            with compact_scope():
                callback(active)
            if not output_called or not captured or row['logical_input_calls'] != len(inputs):
                raise ValueError('The complete segment must consume all declared inputs and produce an output')
            if active.stats()['boots'] != plan['boots']:
                raise ValueError('Actual original Refresh bootstrap count differs from the segment plan')
            outputs, refreshed = [], []
            for position, (before, raw) in enumerate(zip(declared, captured)):
                node = active._plan.outputs.get(position)
                selected = before if node is None else active._planned_cache[node]
                if selected.backend is not active or selected.value is not raw:
                    raise ValueError('Capture the actual raw output after original planned output substitutions')
                outputs.append(self._issue(raw, selected.level, 'segment_output', inputs,
                                           segment=name, position=position, output_refresh_node=node))
                if node is not None:
                    refreshed.append(position)
            if len(outputs) != len(captured) or len(declared) != len(captured):
                raise ValueError('Segment output dimensions differ after planned refresh')
            row.update(status='ok', outputs=[value.serial for value in outputs],
                       output_levels=[value.level for value in outputs],
                       output_refresh_positions=refreshed, refresh=active.stats())
            return outputs
        except BaseException as exc:
            row.update(status='error', error=type(exc).__name__ + ': ' + str(exc), refresh=active.stats())
            raise
        finally:
            try:
                for key, value in previous.items():
                    if value is sentinel:
                        delattr(self.backend, key)
                    else:
                        setattr(self.backend, key, value)
                self._physical_unchanged()
            finally:
                # Only raw physical handles escape this segment. Break both
                # output-hook and RValue-cache cycles before another large
                # segment expands its private plan; the caller's dict stays.
                active.__dict__.pop('output', None)
                active._planned_cache.clear()
                active._cache.clear()
                active._plan = None
                self._busy = False

    def finish(self, handles):
        if self.closed or self._busy:
            raise ValueError('Declare the actual final outputs once between segments')
        handles = list(handles)
        for handle in handles:
            self._require(handle)
        self._physical_unchanged()
        self.backend.output([handle.value for handle in handles])
        self.closed = True
        self.metadata.update(final_output_calls=1, final_outputs=[value.serial for value in handles])


def kernel_callbacks(original_embedding, coefficients=None):
    """Original selector/compact embedding callbacks for external plan creation."""
    import numpy as np
    from .qwen_monotone_selector import analytic_coefficients
    expected = analytic_coefficients(np)
    stages = expected if coefficients is None else tuple(coefficients)
    if len(stages) != 52 or any(not np.array_equal(np.asarray(a), b) for a, b in zip(stages, expected)):
        raise ValueError('Retain52original analytic cubic stages with the final half factor')
    matrix = np.asarray(original_embedding)
    if matrix.dtype != np.float64 or matrix.ndim != 2 or min(matrix.shape) < 2 or not np.isfinite(matrix).all():
        raise ValueError('Require the unchanged finite float64 vocabulary embedding')

    def run(active, include_embedding):
        from ..kernels.encrypted_generation import select_maximum
        from ..kernels import compact_linear
        from ..kernels.layout import multi_flat_layout, layout_blocks
        count = len(layout_blocks(multi_flat_layout(active.slots, matrix.shape[0])))
        logits = [active.input(index) for index in range(count)]
        selected = select_maximum(active, logits, matrix.shape[0], bound=32.,
                                  normalized_gap=2.**-27, coefficients=stages)
        outputs = list(selected['one_hot'])
        if include_embedding:
            mixed, _ = compact_linear.embedding_from_selection(active, selected, matrix)
            if len(mixed) != 1:
                raise ValueError('The original token embedding must occupy one physical block')
            outputs.extend(mixed)
        active.output(outputs)

    return (lambda active: run(active, True)), (lambda active: run(active, False))


def run_staged(backend, *, prefill, prefill_plan, first_kernel_plan, terminal_kernel_plan,
               original_embedding, prompt_length, coefficients=None):
    """Compose supplied plans without model loading or a new global DAG."""
    from ..kernels.layout import multi_flat_layout, layout_blocks
    if type(prompt_length) is not int or not 1 <= prompt_length <= 30:
        raise ValueError('The original prefix requires two reserved padding positions')
    if original_embedding.shape[1] > backend.slots//2:
        raise ValueError('One original token embedding must fit one token block')
    first, terminal = kernel_callbacks(original_embedding, coefficients)
    blocks = len(layout_blocks(multi_flat_layout(backend.slots, original_embedding.shape[0])))
    state = Segments(backend)
    try:
        logits0 = state.run('prefill0', state.inputs, prefill_plan, prefill)
        if len(logits0) != blocks:
            raise ValueError('Preserve every original vocabulary output block')
        first_inputs = state.normalize(logits0, 'prefill0-to-selector0')
        selected0 = state.run('selector0-and-embedding', first_inputs, first_kernel_plan, first)
        if len(selected0) != blocks + 1:
            raise ValueError('First selection must retain its actual single embedding handle')
        masks = state.masks(prompt_length+1)
        handoff = state.normalize([selected0[-1], *masks], 'generated-state-to-prefill1')
        second_inputs = list(state.inputs)
        second_inputs[prompt_length], second_inputs[32], second_inputs[33] = handoff
        state.metadata['handoff'] = dict(source_embedding=selected0[-1].serial,
            normalized_embedding=handoff[0].serial, position=prompt_length,
            consumed_embedding=second_inputs[prompt_length] is handoff[0])
        logits1 = state.run('prefill1', second_inputs, prefill_plan, prefill)
        if len(logits1) != blocks:
            raise ValueError('Preserve the second complete vocabulary output')
        final_inputs = state.normalize(logits1, 'prefill1-to-selector1')
        selected1 = state.run('selector1-terminal', final_inputs, terminal_kernel_plan, terminal)
        if len(selected1) != blocks:
            raise ValueError('Terminal selection outputs one complete vocabulary vector')
        state.finish([*selected0[:blocks], *selected1])
        state.metadata.update(status='ok', vocabulary_blocks=blocks, consumed_embeddings=1,
            terminal_unused_embedding_eliminated=True, expected_boundary_bootstraps=2*blocks+3,
            whole_model_verified=False, C2_verified=False, compile_verified=False,
            real_ckks_executions=0, intermediate_decryptions=0)
        if state.metadata['boundary_bootstraps'] != 2*blocks+3:
            raise ValueError('Explicit boundary bootstrap inventory differs')
        return state
    except BaseException:
        state.metadata['status'] = 'error'
        raise
