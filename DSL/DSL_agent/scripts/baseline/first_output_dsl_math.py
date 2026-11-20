"""Mathematical interpretation of frozen DSL, separate from encrypted execution.

Only the existing checked AST registrar runs. Candidate Python is never loaded,
compiled or executed. These values implement real arithmetic, never ciphertexts;
this module cannot supply or stand in for CKKS execution evidence.
"""
import numpy as np

from delivery_component import require


class MathematicalUnsupported(ValueError):
    pass


class MathematicalWorkLimit(ValueError):
    pass


class _Work:
    def __init__(self, maximum):
        require(type(maximum) is int and 1 <= maximum <= 2**28, 'Finite mathematical work budget')
        self.maximum, self.used = maximum, 0

    def charge(self, size):
        self.used += size
        if self.used > self.maximum:
            raise MathematicalWorkLimit('Frozen mathematical operation budget exhausted')


class _Value:
    """An immutable mathematical vector; obj only supports registrar alias checks."""
    def __init__(self, value, work):
        value = np.array(value, dtype=np.float64, copy=True).reshape(-1)
        require(value.size and np.isfinite(value).all(), 'Finite mathematical values required')
        value.flags.writeable = False
        self.value, self.work, self.obj = value, work, object()

    def _binary(self, other, operation):
        if type(other) is _Value:
            require(other.work is self.work, 'Mathematical execution identity')
            other = other.value
        self.work.charge(max(self.value.size, np.asarray(other).size))
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            return _Value(operation(self.value, other), self.work)

    def __add__(self, other): return self._binary(other, np.add)
    def __sub__(self, other): return self._binary(other, np.subtract)
    def __mul__(self, other): return self._binary(other, np.multiply)
    def __neg__(self): return self._binary(-1., np.multiply)

    def rotate(self, step):
        require(type(step) is int, 'Static mathematical rotation')
        self.work.charge(self.value.size)
        return _Value(np.roll(self.value, -step), self.work)


class _Arithmetic:
    np = np
    Expr = _Value

    def __init__(self, work): self.work = work

    def resolveType(self, value):
        if type(value) is _Value:
            require(value.work is self.work, 'Mathematical value execution identity')
            return value
        return _Value(value, self.work)

    def func(self, signature):
        # The body is the trusted registrar's AST dispatch closure, never a
        # Python function created from candidate source.
        return lambda trusted_body: trusted_body


def evaluate(source, request, inputs, *, maximum_scalar_operations):
    """Return selected mathematical outputs from this exact DSL's checked AST."""
    require(type(source) is bytes, 'Original DSL bytes required')
    from candidate_contract import request_input_names
    from unified_graph_contract import validate_candidate
    from decorated_functions import register
    from unified_capacity_layout import EXTENDED_ABIS
    candidate = dict(schema=1, request_id=request['request_id'], hecate_source=source.decode('utf-8'))
    validate_candidate(candidate, request)
    if 'upstream_helpers' in request:
        raise MathematicalUnsupported('Independent semantics for this upstream helper profile remain uncertified')
    text, constants = candidate['hecate_source'], request['public_constants']
    if request.get('construction_profile') is not None:
        from unified_public_contract import normalize
        lowered = normalize(text, request)
        text, constants = lowered['source'], lowered['constants']
    names = request_input_names(request)
    period = request['layout']['input_slot_period']
    require(type(inputs) is list and len(inputs) == len(names)
            and all(type(x) is list and len(x) == period for x in inputs), 'Mathematical physical input binding')
    work = _Work(maximum_scalar_operations)
    frontend = _Arithmetic(work)
    layout = request['layout'] if request['layout'].get('execution_abi') in EXTENDED_ABIS else None
    functions, plan = register(text, constants, frontend, request['layout']['output_ciphertexts'],
        input_names=names, arrays=True, starred_calls=True, array_arithmetic=True,
        public_loops=True, scalar_augmented=True, array_mutation=True,
        public_constant_indices=request.get("constant_origins",{}).get("capacity_tensor_packing",{}).get("version")=="capacity-public-tensors-v11",
        slot_period=period, capacity_layout=layout)
    ports = functions['golden'](*(frontend.resolveType(x) for x in inputs))
    if type(ports) is np.ndarray:
        require(ports.ndim == 1, 'Explicit one-axis output storage required')
        ports = list(ports)
    elif type(ports) in (tuple, list):
        ports = list(ports)
    else:
        ports = [ports]
    require(len(ports) == request['layout']['output_ciphertexts']
            and all(type(x) is _Value and x.value.size == period for x in ports), 'Mathematical output port binding')
    result = [float(ports[c].value[s]) for c, s in request['layout']['output_selectors']]
    require(bool(result) and all(np.isfinite(x) for x in result), 'Finite mathematical output cells')
    return dict(values=result, scalar_operations=work.used, candidate_python_executed=False,
                encrypted_execution=False, checked_AST=plan)
